import streamlit as st
from model_logic import DreamFrameEngine
from utils import save_png

st.set_page_config(
    layout="wide",
    page_title="DreamFrame GPT"
)

# Initialize engine safely inside session state to prevent reloading across UI element interactions
if "engine" not in st.session_state:
    st.session_state.engine = DreamFrameEngine()

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
[data-testid="column"] {
    background-color: #161618;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #2d2d31;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.4, 3, 1.4])

# LEFT PANEL: Parameters
with col1:
    st.subheader("Control Panel")
    prompt = st.text_area("Describe your vision...", height=180,
                          placeholder="A retro futuristic coffee shop in Kyoto...")
    model_type = st.selectbox("Model Engine", ["FLUX.1-Schnell"])
    size = st.selectbox("Resolution", ["1024x1024", "1216x832", "832x1216"])
    steps = st.slider("Inference Steps", 4, 50, 4 if "schnell" in model_type.lower() else 28)
    scale = st.slider("Guidance Scale", 0.0, 10.0, 0.0 if "schnell" in model_type.lower() else 3.5)
    generate = st.button("Generate Frame", use_container_width=True, type="primary")

# CENTER PANEL: Main Interactive Display
with col2:
    st.title("DreamFrame GPT")

    if generate and prompt:
        with st.spinner("Executing neural rendering pipelines..."):
            try:
                dims = {
                    "1024x1024": (1024, 1024),
                    "1216x832": (1216, 832),
                    "832x1216": (832, 1216)
                }
                w, h = dims[size]

                image, optimized_prompt = st.session_state.engine.generate(prompt, steps, scale, w, h)

                st.image(image, use_container_width=True)

                with st.expander("✨ View GPT-Optimized Prompt Expansion"):
                    st.caption(optimized_prompt)

                path = save_png(image)
                with open(path, "rb") as f:
                    st.download_button(
                        "Download Masterpiece (PNG)",
                        f,
                        file_name=path.split("/")[-1],
                        mime="image/png",
                        use_container_width=True
                    )

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "image": image,
                    "prompt": prompt
                })

                st.rerun()  # Refresh layout smoothly to push items immediately to history columns

            except Exception as e:
                st.error(f"An engine exception occurred during runtime execution: {str(e)}")

# RIGHT PANEL: Smooth Render History Tracker
with col3:
    st.subheader("Render History")
    if "history" in st.session_state and st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.image(item["image"], use_container_width=True)
            st.caption(f"📝 {item['prompt'][:60]}...")
            st.divider()
    else:
        st.info("No frames generated in this session yet.")