from transformers import pipeline
from kittentts import KittenTTS
import soundfile as sf

"""
10/25
- Output is repeating input, not responding to it
- Generated audio is goofy i cant
"""

# Initialize models
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
tts = KittenTTS("models/kitten_tts_nano/kitten_tts_nano_v0_2.onnx")

if __name__ == "__main__":
    # Convert audio to text
    input_text = stt_pipeline("input_audio/test_audio_fr.m4a")["text"]
    print(f"Input text: {input_text}")
    print("-" * 50)

    # Generate response using LLM pipeline
    prompt = prompt = f"Instruction: Reply in French using simple language.\nInput: {input_text}\nOutput:"
    llm_output = llm_pipeline(prompt, max_new_tokens=100)
    output_text = llm_output[0]["generated_text"]
    print(f"Output text: {output_text}")

    # Convert LLM output to speech
    output_audio = tts.generate(output_text)
    sf.write("output_audio/test_output_fr.wav", output_audio, 24000)
