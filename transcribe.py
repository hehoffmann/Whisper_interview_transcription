import contextlib
import wave
import torch
import numpy as np
import time
import datetime
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from faster_whisper import WhisperModel
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

def convert_time(secs):
        return datetime.timedelta(seconds=round(secs))

def convert_to_wav(input_file):
    if input_file == None:
        raise ValueError("No input file given")
    
    try:
        _, file_ending = os.path.splitext(input_file)
        if file_ending == ".wav":
            return input_file
        output_file = input_file.replace(file_ending, ".wav")
        print(f"Converting {file_ending} to .wav")
        os.system(f'ffmpeg -loglevel 0 -n -i "{input_file}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_file}"')

        return output_file
    except Exception as e:
        raise RuntimeError("Converting file to wav failed")

def transcribe(input_file, whisper_model="base", source_language="de", num_speakers=0, output_folder=None):
    if input_file == None:
        raise ValueError("No input file given")
    
    if not os.path.isfile(input_file):
        raise ValueError("Input file not existing")
    
    if output_folder == None:
        raise ValueError("No output folder given")
    elif not os.path.isdir(output_folder):
        raise ValueError("Output folder not existing")

    input_file = convert_to_wav(input_file)

    # Get audio duration
    try:
        print("---")
        with contextlib.closing(wave.open(input_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"Audio file length: {convert_time(duration)} ({round(duration,2)}s)")
    except Exception as e:
        raise RuntimeError("Failed to determine the length of the audio file")

    time_start = time.time()

    # Transcribe
    try:
        print("---\nStart audio transcription with fast-whisper")

        model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")
        
        options = dict(language=source_language, task="transcribe", beam_size=5, best_of=5)
        
        time_transcription_start = time.time()
        
        segments_raw, info = model.transcribe(audio=input_file, **options)
        
        segments = []
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text
            segments.append(chunk)
        print("Successful")
        print(f"> took {convert_time(time.time()-time_transcription_start)} (total {convert_time(time.time()-time_start)})")
    except Exception as e:
        raise RuntimeError("Audio transcription with fast-whisper failed")

    # Create embedding
    try:
        print("---\nStart embedding creation")

        def segment_embedding(segment):
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = Audio().crop(input_file, clip)
            return embedding_model(waveform[None])
                
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        time_embedding_start = time.time()
        
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print("Successful")
        print(f"> took {convert_time(time.time()-time_embedding_start)} (total {convert_time(time.time()-time_start)})")
    except Exception as e:
        raise RuntimeError("Embedding creation failed")
    
    # Assign speaker label and create output
    try:
        print("---\nStart assigning speaker labels")
        time_speaker_start = time.time()

        if num_speakers == 0:
            # Find the best number of speakers
            print("- Start identifying number of speakers")
            
            score_num_speakers = {}
    
            for num_speakers in range(2, 10+1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            num_speakers = max(score_num_speakers, key=lambda x:score_num_speakers[x])

            print(f"- Number of speakers determined: {num_speakers} with {round(score_num_speakers[num_speakers],3)} score")
        else:
            print(f"- Number of speakers given: {num_speakers}")

        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
        
        objects = {'Start' : [], 'End': [], 'Speaker': [], 'Text': []}
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)

        print("Successful")
        print(f"> took {convert_time(time.time()-time_speaker_start)} (total {convert_time(time.time()-time_start)})")
    except Exception as e:
        raise RuntimeError("Assigning speaker labels failed")
    
    # Save output
    output_file = os.path.splitext(os.path.basename(input_file))[0] + '.csv'
    output_path = os.path.join(output_folder, output_file)
    pd.DataFrame(objects).to_csv(output_path)
    print("---\nOutput has been saved: " + output_path)

    return output_path

transcribe("input_folder/input_file.mp4", num_speakers=2, source_language="de", output_folder='output_folder')
