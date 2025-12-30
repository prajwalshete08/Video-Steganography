import streamlit as st
import cv2
import numpy as np
import os
import shutil
import tempfile
import math
from subprocess import call, STDOUT
from datetime import datetime
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import pywt
import matplotlib.pyplot as plt

# Try importing SSIM, handle if missing
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    st.error("Please install scikit-image: pip install scikit-image")
    ssim = None

# --- CONFIGURATION ---
DELTA = 100.0  # High robustness
MAGIC_TAG = b'STEG' 
RSA_CIPHER_BITS = 2048
NONCE_TAG_LENGTH = 32
EMBED_FRAME_INDEX = 1
RSA_KEY_BYTES = RSA_CIPHER_BITS // 8
FRAME_INDEX_FIELD_BYTES = 4
METADATA_FRAME_CAPACITY_BYTES = (
    len(MAGIC_TAG) + 2 + RSA_KEY_BYTES + 4 + FRAME_INDEX_FIELD_BYTES + NONCE_TAG_LENGTH 
)
METADATA_FRAME_BIT_LENGTH = METADATA_FRAME_CAPACITY_BYTES * 8

st.set_page_config(page_title="Robust Video Steganography", page_icon="🛡️", layout="wide")

# --- QUALITY ANALYSIS FUNCTIONS ---

def calculate_metrics(imageA, imageB):
    # 1. MSE (Mean Squared Error)
    mse = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    
    # 2. PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = 100.0
    else:
        pixel_max = 255.0
        psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
        
    # 3. SSIM (Structural Similarity)
    if ssim:
        # Convert to grayscale for SSIM as per standard
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(grayA, grayB, full=True)
    else:
        score = 0.0
        
    return mse, psnr, score

def plot_histogram_comparison(imageA, imageB):
    """
    Generates a Matplotlib figure comparing color histograms.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ('b', 'g', 'r')
    
    for i, color in enumerate(colors):
        # Calculate Histograms
        hist_orig = cv2.calcHist([imageA], [i], None, [256], [0, 256])
        hist_stego = cv2.calcHist([imageB], [i], None, [256], [0, 256])
        
        # Plot
        ax.plot(hist_orig, color=color, linestyle=':', alpha=0.6, label=f'Orig {color.upper()}')
        ax.plot(hist_stego, color=color, linestyle='-', linewidth=1, label=f'Stego {color.upper()}')
    
    ax.set_title("Pixel Intensity Distribution (Original vs Stego)")
    ax.set_xlabel("Pixel Value (0-255)")
    ax.set_ylabel("Frequency")
    # Only show legend for one channel to keep it clean, or simplistic legend
    ax.legend(loc='upper right', fontsize='small', ncol=3)
    plt.tight_layout()
    return fig

# --- HELPER FUNCTIONS ---

def get_downloads_folder():
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.exists(downloads_path): os.makedirs(downloads_path)
    return downloads_path

def load_public_key(key_bytes): return RSA.import_key(key_bytes)
def load_private_key(key_bytes): return RSA.import_key(key_bytes)

def rsa_encrypt(message_bytes, public_key_bytes):
    public_key = load_public_key(public_key_bytes)
    rsa_object = PKCS1_OAEP.new(public_key)
    return rsa_object.encrypt(message_bytes)

def rsa_decrypt(cipher_bytes, private_key_bytes):
    private_key = load_private_key(private_key_bytes)
    rsa_object = PKCS1_OAEP.new(private_key)
    return rsa_object.decrypt(cipher_bytes)

def aes_encrypt(plaintext: str):
    plaintext_bytes = plaintext.encode('utf-8')
    aes_key = get_random_bytes(32)
    cipher = AES.new(aes_key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext_bytes)
    return aes_key, cipher.nonce, tag, ciphertext

def aes_decrypt(aes_key, nonce, tag, ciphertext):
    try:
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
        return decrypted_bytes.decode('utf-8')
    except: return None

def message_to_binary(message):
    if isinstance(message, str): message = message.encode('utf-8')
    return ''.join([bin(byte)[2:].zfill(8) for byte in message])

def binary_to_bytes(binary_string):
    byte_list = []
    for i in range(0, len(binary_string), 8):
        byte_chunk = binary_string[i:i+8]
        if len(byte_chunk) < 8: byte_chunk = byte_chunk.ljust(8, '0')
        byte_list.append(int(byte_chunk, 2))
    return bytes(byte_list)

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def frame_extraction(video, temp_folder):
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    vidcap = cv2.VideoCapture(video)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success: break
        cv2.imwrite(os.path.join(temp_folder, "{:d}.png".format(count)), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count += 1
    vidcap.release()
    return count

# --- DWT + QIM LOGIC ---

def embed_data_qim(frame_path, binary_data):
    frame = cv2.imread(frame_path)
    if frame is None: return False
    
    total_bits = len(binary_data)
    if total_bits == 0: return True
    
    data_idx = 0
    # Green(1) -> Red(2) -> Blue(0)
    for channel in [1, 2, 0]:
        if data_idx >= total_bits: break
        
        channel_data = frame[:, :, channel].astype(np.float64)
        coeffs = pywt.dwt2(channel_data, 'haar')
        LL, (LH, HL, HH) = coeffs
        LL_flat = LL.flatten()
        
        start_idx = 4 
        available_slots = len(LL_flat) - start_idx
        bits_needed = total_bits - data_idx
        bits_to_embed = min(bits_needed, available_slots)
        
        if bits_to_embed > 0:
            indices = slice(start_idx, start_idx + bits_to_embed)
            coeffs_slice = LL_flat[indices]
            bits_chunk = np.array([int(b) for b in binary_data[data_idx : data_idx + bits_to_embed]])
            
            # QIM
            quantized = np.round(coeffs_slice / DELTA)
            is_odd = (quantized % 2).astype(bool)
            target_is_odd = bits_chunk.astype(bool)
            mask_change = (is_odd != target_is_odd)
            quantized[mask_change] += 1.0
            new_coeffs = quantized * DELTA
            
            LL_flat[indices] = new_coeffs
            data_idx += bits_to_embed
            
        LL = LL_flat.reshape(LL.shape)
        coeffs_reconstructed = (LL, (LH, HL, HH))
        channel_reconstructed = pywt.idwt2(coeffs_reconstructed, 'haar')
        frame[:, :, channel] = np.clip(channel_reconstructed, 0, 255)

    frame = frame.astype(np.uint8)
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return True

def extract_data_qim(frame_path, required_bits):
    frame = cv2.imread(frame_path)
    if frame is None or required_bits <= 0: return ""
    
    bits_collected = []
    total_collected_count = 0
    
    for channel in [1, 2, 0]:
        if total_collected_count >= required_bits: break
        channel_data = frame[:, :, channel].astype(np.float64)
        coeffs = pywt.dwt2(channel_data, 'haar')
        LL, _ = coeffs
        LL_flat = LL.flatten()
        
        start_idx = 4
        available_slots = len(LL_flat) - start_idx
        bits_needed = required_bits - total_collected_count
        bits_to_read = min(bits_needed, available_slots)
        
        if bits_to_read > 0:
            indices = slice(start_idx, start_idx + bits_to_read)
            coeffs_slice = LL_flat[indices]
            
            quantized = np.round(coeffs_slice / DELTA)
            extracted_chunk = (quantized % 2).astype(int)
            bits_collected.extend(extracted_chunk)
            total_collected_count += bits_to_read

    return "".join(map(str, bits_collected[:required_bits]))

# --- STREAMLIT WRAPPERS ---

def encode_video_streamlit(video_file, text_content, encode_type, public_key_bytes, temp_dir, filename_to_encode="file.txt"):
    try:
        # 1. Setup Files
        video_ext = os.path.splitext(video_file.name)[1] if hasattr(video_file, 'name') else '.avi'
        video_path = os.path.join(temp_dir, f"input_video{video_ext}")
        with open(video_path, "wb") as f: f.write(video_file.read())
        
        # 2. Encryption
        if encode_type == "Text": TEXT_TO_ENCODE = "TYPE:TEXT|" + text_content
        else: TEXT_TO_ENCODE = f"TYPE:FILE|{filename_to_encode}|{text_content}"
        
        aes_key, nonce, tag, ciphertext = aes_encrypt(TEXT_TO_ENCODE)
        encrypted_aes_key = rsa_encrypt(aes_key, public_key_bytes)
        binary_message = message_to_binary(ciphertext)
        total_message_bits = len(binary_message)

        # 3. Frame Extraction
        frame_folder = os.path.join(temp_dir, "frames")
        total_frames = count_frames(video_path)
        frame_extraction(video_path, frame_folder)
        
        if total_frames <= EMBED_FRAME_INDEX:
            return None, None, f"Not enough frames. Need {EMBED_FRAME_INDEX + 1}, have {total_frames}"

        # 4. Prepare Metadata
        metadata_buffer = bytearray(METADATA_FRAME_CAPACITY_BYTES)
        idx = 0
        metadata_buffer[idx:idx+len(MAGIC_TAG)] = MAGIC_TAG; idx += len(MAGIC_TAG)
        metadata_buffer[idx:idx+2] = len(encrypted_aes_key).to_bytes(2, 'big'); idx += 2
        metadata_buffer[idx:idx+RSA_KEY_BYTES] = encrypted_aes_key.ljust(RSA_KEY_BYTES, b'\x00'); idx += RSA_KEY_BYTES
        metadata_buffer[idx:idx+4] = total_message_bits.to_bytes(4, 'big'); idx += 4
        metadata_buffer[idx:idx+FRAME_INDEX_FIELD_BYTES] = EMBED_FRAME_INDEX.to_bytes(FRAME_INDEX_FIELD_BYTES, 'big'); idx += FRAME_INDEX_FIELD_BYTES
        metadata_buffer[idx:idx+16] = nonce; idx += 16
        metadata_buffer[idx:idx+16] = tag
        
        binary_metadata = message_to_binary(bytes(metadata_buffer))
        
        # 5. Embed Metadata (Frame 0)
        embed_data_qim(os.path.join(frame_folder, "0.png"), binary_metadata)
        
        # 6. Embed Payload & ANALYSIS (Frame 1)
        target_frame_path = os.path.join(frame_folder, f"{EMBED_FRAME_INDEX}.png")
        
        # CACHE ORIGINAL FRAME
        original_frame_cache = cv2.imread(target_frame_path) 
        
        # EMBED (Modifies disk file)
        if not embed_data_qim(target_frame_path, binary_message):
             return None, None, "Embedding failed"
        
        # RELOAD STEGO FRAME
        stego_frame_cache = cv2.imread(target_frame_path)
        
        # CALCULATE METRICS
        mse, psnr, ssim_val = calculate_metrics(original_frame_cache, stego_frame_cache)
        
        # GENERATE PLOT
        hist_fig = plot_histogram_comparison(original_frame_cache, stego_frame_cache)
        
        # 7. Reassemble Video
        final_output_path = os.path.join(temp_dir, f"stego_video_final.avi")
        if shutil.which("ffmpeg") is None: return None, None, "FFmpeg not found."
        
        try:
            call(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", os.path.join(temp_dir, "audio.mp3"), "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
            call(["ffmpeg", "-f", "image2", "-i", os.path.join(frame_folder, "%d.png"), 
                  "-vcodec", "ffv1", "-pix_fmt", "bgr0",  # CRITICAL
                  os.path.join(temp_dir, "video_no_audio.avi"), "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
            call(["ffmpeg", "-i", os.path.join(temp_dir, "video_no_audio.avi"), "-i", os.path.join(temp_dir, "audio.mp3"), 
                  "-codec", "copy", final_output_path, "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
        except Exception as e: return None, None, f"FFmpeg error: {e}"
        
        analysis = {
            "total_frames": total_frames,
            "total_message_bits": total_message_bits,
            "mse": round(mse, 6),
            "psnr": round(psnr, 2),
            "ssim": round(ssim_val, 4),
            "hist_fig": hist_fig  # Store figure object
        }
        return final_output_path, analysis, None
        
    except Exception as e: return None, None, f"Error: {str(e)}"

def decode_video_streamlit(video_file, private_key_bytes, temp_dir):
    try:
        video_ext = os.path.splitext(video_file.name)[1] if hasattr(video_file, 'name') else '.avi'
        video_path = os.path.join(temp_dir, f"stego_video{video_ext}")
        with open(video_path, "wb") as f: f.write(video_file.read())
        
        frame_folder = os.path.join(temp_dir, "decode_frames")
        if not os.path.exists(frame_folder): os.makedirs(frame_folder)
        
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            success, image = cap.read()
            if not success: break
            cv2.imwrite(os.path.join(frame_folder, "{:d}.png".format(count)), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            count += 1
        cap.release()
        
        # Extract Metadata
        meta_bits = extract_data_qim(os.path.join(frame_folder, "0.png"), METADATA_FRAME_BIT_LENGTH)
        if len(meta_bits) < METADATA_FRAME_BIT_LENGTH: return None, "Metadata incomplete"
        
        meta_bytes = binary_to_bytes(meta_bits)
        idx = 0
        
        tag = meta_bytes[idx:idx+len(MAGIC_TAG)]; idx += len(MAGIC_TAG)
        if tag != MAGIC_TAG: return None, f"Signature Mismatch: {tag}"
        
        key_len = int.from_bytes(meta_bytes[idx:idx+2], 'big'); idx += 2
        enc_key = meta_bytes[idx:idx+RSA_KEY_BYTES]; idx += RSA_KEY_BYTES
        msg_bits = int.from_bytes(meta_bytes[idx:idx+4], 'big'); idx += 4
        frame_idx = int.from_bytes(meta_bytes[idx:idx+FRAME_INDEX_FIELD_BYTES], 'big'); idx += FRAME_INDEX_FIELD_BYTES
        nonce = meta_bytes[idx:idx+16]; idx += 16
        auth_tag = meta_bytes[idx:idx+16]
        
        try: aes_key = rsa_decrypt(enc_key, private_key_bytes)
        except: return None, "RSA Decrypt failed"
        
        # Extract Payload
        payload_bits = extract_data_qim(os.path.join(frame_folder, f"{frame_idx}.png"), msg_bits)
        ciphertext = binary_to_bytes(payload_bits)
        
        plaintext = aes_decrypt(aes_key, nonce, auth_tag, ciphertext)
        if not plaintext: return None, "AES Auth failed"
        
        plaintext = plaintext.strip('\x00').strip()
        if plaintext.startswith("TYPE:TEXT|"): return plaintext[10:], "text"
        elif plaintext.startswith("TYPE:FILE|"):
            parts = plaintext.split('|', 2)
            return (parts[2], parts[1]) if len(parts) == 3 else (plaintext, "raw")
        else: return plaintext, "raw"
        
    except Exception as e: return None, f"Error: {str(e)}"

# --- MAIN UI ---

# --- MAIN UI ---

def main():
    st.title("🛡️ Robust Video Steganography with Analysis")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Choose Operation", ["Encode", "Decode", "History"])
    
    if page == "Encode":
        st.header("📝 Encode Data into Video")
        
        # Input Section
        c1, c2 = st.columns(2)
        with c1:
            vid = st.file_uploader("Upload Video", type=['avi', 'mp4'])
            pub = st.file_uploader("Upload Public Key", type=['pem'])
        with c2:
            etype = st.radio("Input Type", ["Text", "File"])
            txt = ""
            fname = "file.txt"
            
            if etype == "Text":
                txt = st.text_area("Enter Text")
            else:
                f = st.file_uploader("Upload File", type=['txt'])
                if f: 
                    txt = f.read().decode('utf-8')
                    fname = f.name
                    
        # Action Button
        if st.button("🚀 Encode & Analyze", type="primary"):
            if vid and pub and txt:
                with st.spinner("Processing: Encryption -> DWT Embedding -> Quality Analysis..."):
                    tmp = tempfile.mkdtemp()
                    try:
                        # Reset pointers
                        vid.seek(0)
                        pub.seek(0)
                        
                        # Run Pipeline
                        res, analysis, err = encode_video_streamlit(vid, txt, etype, pub.read(), tmp, fname)
                        
                        if res:
                            # 1. Save File
                            timestamp = datetime.now().strftime("%H%M%S")
                            dpath = os.path.join(get_downloads_folder(), f"stego_{timestamp}.avi")
                            shutil.copy2(res, dpath)
                            
                            # 2. Show Success
                            st.success(f"✅ Encoding Successful! Saved to: `{dpath}`")
                            
                            # 3. Show Analysis IMMEDIATELY (On the same page)
                            st.markdown("---")
                            st.subheader("📊 Quality Analysis Report")
                            
                            # Metrics Row
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("PSNR", f"{analysis['psnr']} dB", delta="> 35 dB is Good",
                                         help="Peak Signal-to-Noise Ratio. Higher is better.")
                            with m2:
                                st.metric("SSIM", f"{analysis['ssim']}", delta="Target: 1.0",
                                         help="Structural Similarity Index. 1.0 means identical to original.")
                            with m3:
                                st.metric("MSE", f"{analysis['mse']}", help="Mean Squared Error. Lower is better.")
                            with m4:
                                st.metric("Data Size", f"{analysis['total_message_bits']} bits")
                                
                            # Graphs Row
                            st.markdown("##### 📉 Visual Integrity Check")
                            g1, g2 = st.columns([2, 1])
                            
                            with g1:
                                # Show Histogram
                                if 'hist_fig' in analysis:
                                    st.pyplot(analysis['hist_fig'])
                                    
                                    st.caption("Color Histogram: Overlapping lines indicate high stealth.")
                                else:
                                    st.warning("Histogram data unavailable.")
                            
                            with g2:
                                # Show Interpretation Guide
                                st.info("""
                                **Quick Guide:**
                                * **PSNR > 40dB**: Excellent (Invisible)
                                * **PSNR 30-40dB**: Good (Hard to see)
                                * **SSIM > 0.95**: Structural integrity preserved.
                                """)
                                
                            # Save to session state for History tab
                            st.session_state.last_analysis = analysis
                            
                        else:
                            st.error(f"❌ Encoding Failed: {err}")
                    finally:
                        shutil.rmtree(tmp)
            else:
                st.warning("⚠️ Please upload all required files (Video, Key, and Data).")

    elif page == "Decode":
        st.header("🔍 Decode Data")
        vid = st.file_uploader("Upload Stego Video", type=['avi'])
        priv = st.file_uploader("Upload Private Key", type=['pem'])
        
        if st.button("🔓 Decode", type="primary"):
            if vid and priv:
                with st.spinner("Decoding..."):
                    tmp = tempfile.mkdtemp()
                    try:
                        vid.seek(0)
                        priv.seek(0)
                        res, meta = decode_video_streamlit(vid, priv.read(), tmp)
                        
                        if res:
                            st.success("✅ Decoding Successful!")
                            st.markdown("---")
                            
                            if meta == "text":
                                st.subheader("📜 Decoded Text")
                                st.text_area("Content", res, height=200)
                            elif meta == "raw":
                                st.subheader("⚠️ Raw Output (Unknown Format)")
                                st.text_area("Content", res)
                            else:
                                st.subheader(f"📁 Decoded File: {meta}")
                                st.download_button(f"⬇️ Download {meta}", res, meta)
                        else:
                            st.error(f"❌ Decoding Failed: {meta}")
                    finally:
                        shutil.rmtree(tmp)
            else:
                st.warning("⚠️ Please upload Video and Private Key.")

    elif page == "History":
        st.header("📜 Session History")
        if 'last_analysis' in st.session_state:
            st.json(st.session_state.last_analysis)
        else:
            st.info("No operations performed in this session yet.")

if __name__ == "__main__":
    main()