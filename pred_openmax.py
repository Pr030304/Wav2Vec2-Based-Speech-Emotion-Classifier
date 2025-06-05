## Complete Code for Inferencing

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa
import numpy as np
import torch.nn.functional as F
import pickle
from scipy.stats import weibull_min
from scipy.special import softmax
import random
from scipy.spatial.distance import cdist

T = 1.536

with open("openmax_params_new.pkl", "rb") as f:
    data = pickle.load(f)
class_means    = data["class_means"]
class_cov_invs = data["class_cov_invs"]
class_weibulls = data["class_weibulls"]

inverse_label_map = {0: 'surprise', 1: 'angry', 2: 'neutral', 3: 'sad', 4: 'happy'}

# Configuration
NEW_CLASS_THRESHOLD = 5.0    # distance threshold in embedding space
DISTANCE_METRIC      = 'euclidean'  # or 'cosine'

# State (persist across inferences; initialize once)
novel_prototypes = {}   # {new_class_id: centroid_embedding (np.ndarray)}
novel_counts     = {}   # {new_class_id: number_of_assigned_samples}
next_novel_id    = 0


def get_input(audio_path, processor):
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
            if(len(speech_part) <= 0.8*max_length and i == k-1 and k != 1):                                  
                continue
            if(len(speech_part) < max_length):                                        # padding the speech part if its length is less than max_length                 
                speech_part = np.pad(speech_part, (0, max_length - len(speech_part)), 'constant')
            speeches.append(speech_part)

    input_lis = []
    for speech in speeches:
        inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, max_length=max_length)
        input_values = inputs.input_values.squeeze()
        input_lis.append(input_values)

    return input_lis

def openmax_probs(input_values, model, T=1.0):
    with torch.no_grad():
        # a) penultimate AV: mean-pooled encoder hidden states
        input_values = input_values.unsqueeze(0)
        hidden = model.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        av = hidden.cpu().numpy().squeeze()
        av_norm = av / (np.linalg.norm(av) + 1e-12)

        # b) raw logits
        logits = model(input_values).logits.cpu().numpy().squeeze()

    # c) recalibration
    recal = []
    unk_mass = 0.0
    K = model.config.num_labels
    for c in range(K):
        mu      = class_means[c]
        invcov  = class_cov_invs[c]
        delta   = av_norm - mu
        dist    = np.sqrt(delta @ invcov @ delta)
        shape, loc, scale = class_weibulls[c]
        w       = weibull_min.cdf(dist, shape, loc=loc, scale=scale)
        alpha   = 1.0 - w
        recal.append(alpha * logits[c])
        unk_mass += (1.0 - alpha) * logits[c]

    openmax_logits = np.append(recal, unk_mass)
    return softmax(openmax_logits / T)

def get_embedding(input_values, model):
    """
    Returns the mean‑pooled Wav2Vec2 hidden embedding (1, H) as a 1D numpy array.
    """
    input_values = input_values.unsqueeze(0)  # (1, T)
    model.eval()
    with torch.no_grad():
        hidden = model.wav2vec2(input_values).last_hidden_state  # (1, T, H)
    emb = hidden.mean(dim=1).cpu().numpy().squeeze()             # (H,)
    return emb

def assign_or_create_novel(emb: np.ndarray):
    """
    emb: 1D numpy array of shape (H,)
    Returns: class_label (int), where
             0-4 are known classes, and ≥5 are dynamic novel IDs.
    """
    global next_novel_id

    # 1) Check distance to each existing novel prototype
    if novel_prototypes:
        centroids = np.stack(list(novel_prototypes.values()))  # (M, H)
        dists = cdist(centroids, emb.reshape(1, -1), metric=DISTANCE_METRIC)  # (M,)
        print(dists)
        idx_min = dists.argmin()
        if dists[idx_min] <= NEW_CLASS_THRESHOLD:
            # assign to that novel class
            novel_id = list(novel_prototypes.keys())[idx_min]
            # update centroid incrementally:
            count = novel_counts[novel_id]
            new_centroid = (novel_prototypes[novel_id]*count + emb) / (count + 1)
            novel_prototypes[novel_id] = new_centroid
            novel_counts[novel_id] += 1
            return novel_id

    # 2) Otherwise, create a brand‑new novel class
    novel_id = 5 + next_novel_id
    novel_prototypes[novel_id] = emb.copy()
    novel_counts[novel_id]     = 1
    next_novel_id += 1
    return novel_id

def inference_function(input_lis, model):
    probs = []
    for input_values in input_lis:
        prob = openmax_probs(input_values, model, T)
        probs.append(prob)
    probs = np.mean(probs, axis=0)  # shape (K+1,)

    known_probs = probs[:-1]               # P over your 5 emotions
    unk_prob    = probs[-1]                # P(unknown)

    # decide label
    if unk_prob > 0.1:                             # choose your threshold
        emb = get_embedding(input_values, model)
        novel_label_id = assign_or_create_novel(emb)
        label = f"unknown_{novel_label_id - 4}"
    else:
        pred = known_probs.argmax()
        label = inverse_label_map[pred]
    
    return label, probs.tolist(), known_probs.tolist()


# # # Load the processor and model from the checkpoint directory
model_new = Wav2Vec2ForSequenceClassification.from_pretrained("checkpoint-5250", output_hidden_states=True)
processor_new = Wav2Vec2Processor.from_pretrained("checkpoint-processor")
# audio_path = "white-noise-171891.mp3"

# inputs = get_input(audio_path, processor_new)

# predictions, total_prob, known_prob = inference_function(inputs, model_new)

# print('Predicted Label:', predictions)
# print('Probabilities : ', total_prob)
# print('Known Probabilities : ', known_prob)
# print('---------------------------------------------------------------------------------------')

audio_path = "/Users/srijananand/Documents/ee698r/project/wav2vec2_approach/new_model/whysoserious.wav"

inputs = get_input(audio_path, processor_new)

predictions, total_prob, known_prob = inference_function(inputs, model_new)

print('Predicted Label:', predictions)
print('Probabilities : ', total_prob)
print('Known Probabilities : ', known_prob)
print('---------------------------------------------------------------------------------------')

audio_path = "/Users/srijananand/Documents/ee698r/project/wav2vec2_approach/new_model/whysoserious.wav"

inputs = get_input(audio_path, processor_new)

predictions, total_prob, known_prob = inference_function(inputs, model_new)

print('Predicted Label:', predictions)
print('Probabilities : ', total_prob)
print('Known Probabilities : ', known_prob)
print('---------------------------------------------------------------------------------------')

# audio_path = "DC_a14.wav"

# inputs = get_input(audio_path, processor_new)

# predictions, total_prob, known_prob = inference_function(inputs, model_new)

# print('Predicted Label:', predictions)
# print('Probabilities : ', total_prob)
# print('Known Probabilities : ', known_prob)
# print('---------------------------------------------------------------------------------------')

# audio_path = "DC_a14.wav"

# inputs = get_input(audio_path, processor_new)

# predictions, total_prob, known_prob = inference_function(inputs, model_new)

# print('Predicted Label:', predictions)
# print('Probabilities : ', total_prob)
# print('Known Probabilities : ', known_prob)
# print('---------------------------------------------------------------------------------------')