## Complete Code for Inferencing

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa
import numpy as np
import torch.nn.functional as F
import pickle


inverse_label_map = {0: 'surprise', 1: 'angry', 2: 'neutral', 3: 'sad', 4: 'happy'}

def inference_function(model, processor, audio_path, optimal_T = 1.536, H_thresh = 0.75, COSINE_THRESH = 0.85, clusters = None):

    # Switch model to evaluation mode
    if clusters is None:
        clusters = []
    model.eval()

    # load the audio file with sampling rate = 16000 (as per the facebook model)
    speech, sr = librosa.load(audio_path, sr = 16000)

    # pad or truncate the speech to the required length
    speeches = []
    max_length = 40000
    if len(speech) < max_length:
        speech = np.pad(speech, (0, max_length - len(speech)), 'constant')
        speeches.append(speech)
    else:
        k = (len(speech) + max_length - 1) // max_length
        for i in range(k):
            if (i == k-1):
                speech_part = speech[i*max_length:]
            else:
                speech_part = speech[i*max_length : (i+1)*max_length]
            if(len(speech_part) < max_length):                                        # padding the speech part if its length is less than max_length                 
                speech_part = np.pad(speech_part, (0, max_length - len(speech_part)), 'constant')
            if(len(speech_part) <= 0.7*max_length):                                   # ignoring the speech part if its length is less than 70% of max_length
                continue
            speeches.append(speech_part)

    input_lis = []
    for speech in speeches:
        inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, max_length=max_length)
        input_values = inputs.input_values.squeeze()
        input_lis.append(input_values)

    preds = []
    states = []

    with torch.no_grad():
        for input_values in input_lis:
            outputs = model(input_values.unsqueeze(0))
            new_logits = outputs.logits
            h_states = outputs.hidden_states
            new_logits_scaled = new_logits / optimal_T
            probabilities = F.softmax(new_logits_scaled, dim=-1)
            preds.append(probabilities)
            states.append(h_states[-1])

    last_layer = torch.mean(torch.stack(states), dim=0)
    feat = last_layer.mean(dim=1)[0] # (hidden_size,)

    probabilities = torch.mean(torch.stack(preds), dim=0)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=1)

    outp = ""

    if(entropy <= H_thresh):
        predictions = probabilities.argmax(dim=-1).item()
        outp = inverse_label_map[predictions]
    else:
        if clusters:
            sims = [
                F.cosine_similarity(feat, c["center"], dim=0).item()
                for c in clusters
            ]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            if sims[best_idx] >= COSINE_THRESH:
                c = clusters[best_idx]
                c["center"] = (c["center"]*c["count"] + feat) / (c["count"]+1)
                c["count"] += 1
                outp = f"unknown_type_{best_idx+1}"
        
        if outp == "":
            clusters.append({"center": feat.clone(), "count": 1})
            outp = f"unknown_type_{len(clusters)}"
        
    return [clusters, outp, probabilities.tolist()[0]]




# # # Load the processor and model from the checkpoint directory
# model_new = Wav2Vec2ForSequenceClassification.from_pretrained("checkpoint-5250", output_hidden_states=True)
# processor_new = Wav2Vec2Processor.from_pretrained("checkpoint-processor")
# audio_path = "whysoserious.wav"

# clusters, predictions, probabilities = inference_function(model_new, processor_new, audio_path)

# print('Predicted Label:', predictions)
# print('Probabilities : ', probabilities)