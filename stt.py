import os
from pydub import AudioSegment
from faster_whisper import WhisperModel
import whisperx
from pyannote.audio import Pipeline
import torch
import numpy as np
import requests
from io import BytesIO
import soundfile as sf

class STT:
    def __init__(self,input_audio_url,num_speakers, device):
        self.input_audio_url = input_audio_url
        self.num_speakers = num_speakers
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.download_audio = self.download_audio_to_memory()
        self.vad_pipeline = self.load_vad_pipeline()
        self.diarization_pipeline = self.load_diarization_pipeline()
        self.whisper_model = self.load_whisper_model()
        self.align_model, self.metadata = self.load_align_model()

    def load_whisper_model(self):
        print("위스퍼_모델_로딩중")
        return WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device='cuda', compute_type='int8')

    def load_align_model(self):
        print("후처리_모델_로딩중")
        return whisperx.load_align_model(language_code='ko', device=self.device)

    def load_vad_pipeline(self):
        print("전처리_모델_로딩중")
        return Pipeline.from_pretrained("pyannote/voice-activity-detection").to(self.device)

    def load_diarization_pipeline(self):
        print("pyannote_로딩중")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        return pipeline.to(torch.device(self.device))

    def download_audio_to_memory(self):
        try:
            response = requests.get(self.input_audio_url, stream=False)
            response.raise_for_status()  # HTTP 요청이 성공적으로 완료되지 않을 경우 예외 발생
            audio_data = BytesIO(response.content)
            print(type(audio_data))
            return audio_data
        except requests.exceptions.RequestException as e:
            print(f"오디오 다운로드 실패: {e}")
            return None

    def fixed_audio(audio):
        temp_audio = 'temp.wav'
        x, _ = librosa.load(audio, sr=16000)
        sf.write(temp_audio, x, 16000)
        return temp_audio

    def perform_vad_on_full_audio(self,audio):
        print("VAD처리중")
        audio = AudioSegment.from_wav(audio)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # 형태를 (채널, 시간) 형식으로 만든다.
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).T
        else:
            samples = samples.reshape(1, -1)

        waveform = torch.from_numpy(samples)

        try:
            vad = self.vad_pipeline({"waveform": waveform, "sample_rate": audio.frame_rate})
        except Exception as e:
            print(f"VAD_실패: {e}")
            return None

        speech_segments = vad.get_timeline().support()
        print(f"{len(speech_segments)}개의 세그먼트")

        speech_audio = AudioSegment.empty()
        for segment in speech_segments:
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            speech_audio += audio[start_ms:end_ms]

        target_sampling_rate =16000
        speech_audio = speech_audio.set_frame_rate(target_sampling_rate).set_channels(1).set_sample_width(2)

        return speech_audio

    def split_audio(self, audio, chunk_duration=600000):
        print("10단위로_나누는_중")
        duration = len(audio)
        chunks = [audio[i:i + chunk_duration] for i in range(0, duration, chunk_duration)]
        print(f"{len(chunks)} 청크_생성중")
        return chunks

    def perform_vad_on_chunk(self, audio_chunk):
        print("VAD_청로 나누기")
        samples = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)

        if audio_chunk.channels == 2:
            samples = samples.reshape(-1, 2).T
        else:
            samples = samples.reshape(1, -1)

        waveform = torch.from_numpy(samples)

        try:
            vad = self.vad_pipeline({"waveform": waveform, "sample_rate": audio_chunk.frame_rate})
        except Exception as e:
            print(f"청크화_실패: {e}")
            return None

        speech_segments = vad.get_timeline().support()
        print(f"{len(speech_segments)}_발화_인식한_세그먼트")

        speech_audio = AudioSegment.empty()
        for segment in speech_segments:
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            speech_audio += audio_chunk[start_ms:end_ms]

        return speech_audio

    def transcribe_audio(self, audio_segment):
        print("전사_처리")
        try:
            audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(audio_array,beam_size=5)
            segments=list(segments)
            print("전사_처리_완료")
            transcription_segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments]
            print(transcription_segments)
            return transcription_segments
        except Exception as e:
            print(f"전사_처리_실패: {e}")
            return None

    def post_process_whisperx(self, transcription_segments, audio_segment):
        print("WhisperX로_후처리_중")
        try:
            audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            result_aligned = whisperx.align(transcription_segments, self.align_model, self.metadata, audio_array, device=self.device)
            aligned_segments = result_aligned.get("segments", []) if isinstance(result_aligned, dict) else []
            print(f"실제 음성 세그먼트 {len(aligned_segments)}")
            return aligned_segments
        except Exception as e:
            print(f"후처리_에러: {e}")
            return None

    def perform_diarization(self, vad_audio):
        print("화자_분리")
        try:
            audio_array = np.array(vad_audio.get_array_of_samples()).astype(np.float32) / 32768.0

            if vad_audio.channels == 2:
                audio_array = audio_array.reshape(-1, 2).T
            else:
                audio_array = audio_array.reshape(1, -1)

            diarization = self.diarization_pipeline({"waveform": torch.from_numpy(audio_array), "sample_rate": vad_audio.frame_rate}, num_speakers=self.num_speakers)
            print("화자분리_완료")
            return diarization
        except Exception as e:
            print(f"화자분리_실패: {e}")
            return None

    def match_speaker_to_segments(self, diarization, transcription_segments):
        print("화자_매칭중")
        matched_segments=[]

        for segment in transcription_segments:
            midpoint = (segment['start'] + segment['end']) / 2
            speaker_found = False
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <=midpoint <= turn.end:
                    matched_segments.append((segment['start'], segment['end'], speaker, segment['text']))
                    speaker_found =True
                    break
                if not speaker_found:
                    matched_segments.append((segment['start'], segment['end'], "Unknown", segment['text']))

        matched_segments.sort(key=lambda x: x[0])
        print("화자_매칭_완료")
        return matched_segments

    def save_transcriptions_as_json(self, matched_segments):
        print("json형태로_변환중")
        try:
            transcriptions = []
            for segment in matched_segments:
                start_time, end_time, speaker, text = segment
                transcriptions.append({
                    "speaker": speaker,
                    "text": text
                })
            content={"minutes":transcriptions}
            return content
        except Exception as e:
            print(f"json형태로_변환실패: {e}")
            return None

    def process_chunk(self, chunk):
        print("청크화_진행중")
        vad_audio = self.perform_vad_on_chunk(chunk)
        if vad_audio is None:
            return None

        transcription_segments = self.transcribe_audio(vad_audio)
        if transcription_segments is None:
            return None

        aligned_segments = self.post_process_whisperx(transcription_segments, vad_audio)
        if aligned_segments is None:
            return None

        return aligned_segments

    def run(self):
        audio_data = self.download_audio()
        if audio_data is None:
            print("오디오_다운로드_실패")
            return
        
        audio_data = self.fixed_audio(audio_data)
        if audio_data is None:
           print("오디오_파일_정상화_실패")
           return

        vad_audio = self.perform_vad_on_full_audio(audio_data)
        if vad_audio is None:
            print("VAD처리_오류")
            return

        diarization = self.perform_diarization(vad_audio)
        if diarization is None:
            print("화자분리_실패")
            return

        chunks = self.split_audio(vad_audio)

        all_aligned_segments = []
        for i, chunk in enumerate(chunks):
            print(f"청크_생성 {i + 1}/{len(chunks)}")
            chunk_segments = self.process_chunk(chunk)
            if chunk_segments:
                chunk_duration = len(chunk) / 1000
                for seg in chunk_segments:
                    seg['start'] += i * chunk_duration
                    seg['end'] += i * chunk_duration
                all_aligned_segments.extend(chunk_segments)

        matched_segments = self.match_speaker_to_segments(diarization, all_aligned_segments)
        content = self.save_transcriptions_as_json(matched_segments)
        print("STT완료")
        return content