import streamlit as st
from model_logic import DreamFrameEngine

# Page ko wide setup karein taake columns achi tarah phail sakein
st.set_page_config(layout="wide", page_title="DreamFrame Pro")
engine = DreamFrameEngine()

# Custom CSS for Dark UI and Columns
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    /* Columns ko thora aur jagah dene ke liye */
    [data-testid="column"] {background-color: #161618; padding: 20px; border-radius: 12px; margin: 5px;}
    </style>
    """, unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.5, 3, 1.5])

# --- LEFT COLUMN: Controls ---
with col_left:
    st.subheader("Prompt")
    prompt = st.text_area("Describe your vision...", height=150)

    st.subheader("Settings")
    model = st.selectbox("Style", ["Fast Mode", "Artistic Mode", "Realistic Mode"])
    size = st.selectbox("Resolution", ["Square (512x512)", "Landscape (768x512)", "Portrait (512x768)"])
    steps = st.slider("Steps", 10, 50, 30)
    scale = st.slider("Scale", 1.0, 20.0, 9.5)

    generate_btn = st.button("Generate Image", type="primary", use_container_width=True)

# --- CENTER COLUMN: Main Image ---
with col_center:
    st.markdown("<h2 style='text-align: center;'>Generated Image</h2>", unsafe_allow_html=True)

    if generate_btn and prompt:
        with st.spinner("Generating..."):
            dims = {"Square (512x512)": (512, 512), "Landscape (768x512)": (768, 512), "Portrait (512x768)": (512, 768)}
            w, h = dims.get(size, (512, 512))
            img = engine.generate(prompt, steps, scale, w, h, model)

            # Yahan width='stretch' use karein
            st.image(img, width='stretch')

            st.info(f"**Prompt:** {prompt}")

            if 'history' not in st.session_state: st.session_state.history = []
            st.session_state.history.append({"img": img, "prompt": prompt})

# --- RIGHT COLUMN: History ---
with col_right:
    st.subheader("🕒 History")
    if 'history' in st.session_state:
        for item in reversed(st.session_state.history):
            # Yahan bhi width='stretch' use karein
            st.image(item['img'], width='stretch')
            st.caption(f"{item['prompt'][:50]}...")
            st.divider()