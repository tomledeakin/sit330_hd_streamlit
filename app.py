import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import timm
import spacy
import numpy as np

# -------------------------
# 1. CONFIGURATION
# -------------------------
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH     = "transformer_vqa_model.pth"
VOCAB_PATH     = "vocab.json"
IDX2LABEL_PATH = "idx2label.json"
SAMPLES_DIR    = "samples"

# -------------------------
# 2. LOAD MAPPINGS
# -------------------------
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
with open(IDX2LABEL_PATH, 'r', encoding='utf-8') as f:
    idx2label_str = json.load(f)
# convert JSON string keys back to int
idx2label = {int(k): v for k, v in idx2label_str.items()}

VOCAB_SIZE = len(vocab)
MAX_LEN    = 20
PAD_IDX    = vocab.get("<pad>", 0)

# -------------------------
# 3. TOKENIZER
# -------------------------
nlp = spacy.blank("en")  # only tokenizer

def tokenize(text):
    return [tok.text.lower() for tok in nlp(text)]

# -------------------------
# 4. ENCODING
# -------------------------
def encode_question(text):
    ids = [vocab.get(w, vocab.get("<unk>")) for w in tokenize(text)]
    ids = ids[:MAX_LEN] + [PAD_IDX] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# -------------------------
# 5. IMAGE TRANSFORMS
# -------------------------
img_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(180),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -------------------------
# 6. MODEL COMPONENTS
# -------------------------
class MultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, L, _ = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(B, L, self.nh, self.dk).transpose(1,2)
        q, k, v = map(split, (q, k, v))
        scores = (q @ k.transpose(-2,-1)) / np.sqrt(self.dk)
        attn   = scores.softmax(dim=-1)
        out    = (attn @ v).transpose(1,2).contiguous().view(B, L, -1)
        return self.proj(out)

class EncoderLayer(nn.Module):
    def __init__(self, d=128, h=8, ff=256, p=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttn(d, h)
        self.drop = nn.Dropout(p)
        self.ln2  = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d=128, depth=2, h=8):
        super().__init__()
        self.tok    = nn.Embedding(vocab_size, d, padding_idx=PAD_IDX)
        self.pos    = nn.Parameter(torch.randn(1, max_len, d))
        self.layers = nn.ModuleList([EncoderLayer(d, h) for _ in range(depth)])
    def forward(self, idx):
        x = self.tok(idx) + self.pos[:, :idx.size(1)]
        for lay in self.layers:
            x = lay(x)
        return x.mean(dim=1)

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True, num_classes=0)
        for p in self.backbone.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.backbone(x)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden=256, n_out=None):
        super().__init__()
        n_out = n_out or len(idx2label)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_out)
        )
    def forward(self, x):
        return self.net(x)

class VQATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.text   = TextTransformer(VOCAB_SIZE, MAX_LEN).to(DEVICE)
        self.vision = VisionEncoder().to(DEVICE)
        self.clf    = Classifier(512+128).to(DEVICE)
    def forward(self, img, q):
        t = self.text(q)
        v = self.vision(img)
        return self.clf(torch.cat([v, t], dim=-1))

# -------------------------
# 7. LOAD MODEL
# -------------------------
model = VQATransformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# 8. STREAMLIT APP WITH TABS
# -------------------------
st.title("VQA Transformer Demo")

tab_upload, tab_samples = st.tabs(["Upload Image", "Sample Images"])

# Tab 1: Upload
with tab_upload:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    question1    = st.text_input("Enter your question:", key="upload_q")
    if uploaded_img:
        img1 = Image.open(uploaded_img).convert("RGB")
        st.image(img1, caption="Uploaded Image", use_container_width=True)
        if question1:
            img_t  = img_transforms(img1).unsqueeze(0).to(DEVICE)
            q_ids  = encode_question(question1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits   = model(img_t, q_ids)
                probs    = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_i   = int(np.argmax(probs))
                pred_lbl = idx2label[pred_i]
            st.write(f"**Question:** {question1}")
            st.write(f"**Answer:** {pred_lbl}")
            st.write("**Top 5:**")
            for i in np.argsort(probs)[-5:][::-1]:
                st.write(f"{idx2label[i]}: {probs[i]:.3f}")

# Tab 2: Samples
with tab_samples:
    samples = []
    if os.path.isdir(SAMPLES_DIR):
        samples = sorted([f for f in os.listdir(SAMPLES_DIR)
                          if f.lower().endswith((".jpg",".jpeg",".png"))])
    choice = st.selectbox("Choose a sample image", ["--"] + samples)
    question2 = st.text_input("Enter your question:", key="sample_q")
    if choice and choice != "--":
        img2 = Image.open(os.path.join(SAMPLES_DIR, choice)).convert("RGB")
        st.image(img2, caption=f"Sample: {choice}", use_container_width=True)
        if question2:
            img_t   = img_transforms(img2).unsqueeze(0).to(DEVICE)
            q_ids   = encode_question(question2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits   = model(img_t, q_ids)
                probs    = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_i   = int(np.argmax(probs))
                pred_lbl = idx2label[pred_i]
            st.write(f"**Question:** {question2}")
            st.write(f"**Answer:** {pred_lbl}")
            st.write("**Top 5:**")
            for i in np.argsort(probs)[-5:][::-1]:
                st.write(f"{idx2label[i]}: {probs[i]:.3f}")