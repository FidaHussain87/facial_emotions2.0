# --- To run the backend (save this as main.py) ---
# You'll need to install uvicorn: pip install uvicorn
# And then run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Required Python packages:
# pip install fastapi uvicorn python-multipart opencv-python deepface tensorflow (or tensorflow-cpu)
#
# Note on TensorFlow:
# DeepFace uses TensorFlow. If you have a GPU and CUDA installed, `tensorflow` will use it.
# Otherwise, `tensorflow-cpu` is a good alternative.
# For macOS M1/M2/M3/M4, you'll need `tensorflow-macos`.
# `pip install tensorflow-macos tensorflow-metal`
#
# DeepFace will download pre-trained models on its first run.
# This might take some time.