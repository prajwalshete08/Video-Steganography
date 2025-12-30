import cv2
import numpy as np
import math
import os
import shutil
import sys
import re
from termcolor import cprint
from pyfiglet import figlet_format
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import unpad
from pathlib import Path

# --- Configuration & Helpers (Must Match Encoder) ---

# FIX 1: Set LSB_DEPTH = 2 to match the mask logic (& 3) and data size (zfill(2))
LSB_DEPTH = 2 
RSA_CIPHER_BITS = 2048 
NONCE_TAG_LENGTH = 32 # 16 bytes Nonce + 16 bytes Tag

# -------------------- Utility Functions --------------------
def get_downloads_folder():
    """Returns the absolute path to the user's Downloads directory."""
    # Method 1: Using Path.home() (Best cross-platform standard)
    downloads_path = Path.home() / "Downloads"
    
    # Ensure the directory exists (it usually does, but this is safer)
    os.makedirs(downloads_path, exist_ok=True)
    
    return str(downloads_path)

def load_private_key(key_path):
    """Loads the user's RSA private key."""
    with open(key_path, 'rb') as f:
        return RSA.import_key(f.read())

def rsa_decrypt(cipher_bytes, key_path):
    """Decrypts the small AES key using RSA Private Key."""
    private_key = load_private_key(key_path)
    rsa_object = PKCS1_OAEP.new(private_key)
    return rsa_object.decrypt(cipher_bytes)

def aes_decrypt(aes_key, nonce, tag, ciphertext):
    """Decrypts large ciphertext using the recovered AES components."""
    try:
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
        return unpad(decrypted_bytes, AES.block_size).decode('utf-8')
    except ValueError:
        cprint("[ERROR] AES decryption failed (Incorrect key/tag/nonce or data corruption).", 'red')
        return None
    except Exception as e:
        cprint(f"[ERROR] An unexpected decryption error occurred: {e}", 'red')
        return None

def binary_to_bytes(binary_string):
    """Converts a binary string to a bytes object, padding the last byte with '0's if necessary."""
    byte_list = []
    
    for i in range(0, len(binary_string), 8):
        byte_chunk = binary_string[i:i+8]
        
        if len(byte_chunk) < 8:
            byte_chunk = byte_chunk.ljust(8, '0') 
            
        byte_list.append(int(byte_chunk, 2))
        
    return bytes(byte_list)

def extract_data_2bit_lsb(frame_path, required_bits):
    """
    Extracts binary data from the frame using 2-bit LSB up to required_bits.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return ""
    
    binary_data = []
    bit_count = 0
    
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            for c in range(3):
                if bit_count >= required_bits: break
                    
                pixel_val = frame[i, j, c]
                lsb_int = pixel_val & 3 # Mask for 2 LSBs
                binary_chunk = bin(lsb_int)[2:].zfill(2) # Output 2 bits
                
                binary_data.append(binary_chunk)
                # FIX 2: Bit count must increment by 2
                bit_count += 2
            
            if bit_count >= required_bits: break
        if bit_count >= required_bits: break

    result_binary = "".join(binary_data)
    return result_binary[:required_bits]

def clean_tmp(path="./tmp_decode"):
    """Removes the temporary folder and its contents."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            cprint(f"[INFO] Temporary files at '{path}' are cleaned up.", 'yellow')
        except OSError as e:
            cprint(f"[ERROR] Error removing directory {path}: {e}", 'red')

# -------------------- Main Decoder Function --------------------

def decode_video_data(video_path):
    cprint(figlet_format('Video Decoder', font='digital'),'yellow', attrs=['bold']) 
    
    private_key_path = input("Enter your RSA private key filename with path (.pem): ")
    external_stego_path = input("Enter the path to the external image containing the frame map: ")
    
    external_stego_path = external_stego_path.strip().strip('"').strip("'")
    temp_folder="./tmp_decode"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    

    
    # 1. Frame Extraction 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cprint("[INFO] Extracting frames for analysis...", 'blue')
    count = 0
    while True:
        success, image = cap.read()
        if not success: break
        cv2.imwrite(os.path.join(temp_folder, "{:d}.png".format(count)), image)
        count += 1
    cprint("[SUCCESS] All frames extracted to './tmp_decode'", 'green')

    # --- STEP 1: Recover Key and Frame Map from External Image ---
    cprint("\n[STEP 1] Recovering Metadata...", 'cyan')
    try:
        from stegano import lsb 
        extracted_metadata = lsb.reveal(external_stego_path)
    except Exception as e:
        cprint(f"[ERROR] Failed to extract map from external image: {e}", 'red'); return

    # Expected format: EncryptedKeyHEX | TotalCiphertextBits | [Frame List]
    try:
        parts = extracted_metadata.split('|', 2)
        if len(parts) != 3: raise ValueError("Metadata split error.")
            
        encrypted_key_hex = parts[0]
        total_message_bits = int(parts[1])
        frame_map_string = parts[2]
    except Exception:
        cprint("[ERROR] Extracted metadata is corrupted or not in the expected format.", 'red'); return
        
    match = re.search(r'\[.*\]', frame_map_string)
    if not match: cprint("[ERROR] Could not parse frame list from extracted metadata.", 'red'); return
    selected_frames = eval(match.group(0))
        
    cprint(f"[SUCCESS] Recovered {len(selected_frames)} embedding frame indices. Total message bits: {total_message_bits}", 'green')

    # --- STEP 2: Decrypt AES Key ---
    cprint("\n[STEP 2] Decrypting AES Key...", 'cyan')
    if len(encrypted_key_hex) != (RSA_CIPHER_BITS // 4): 
        cprint(f"[FATAL ERROR] Encrypted key length mismatch.", 'red'); return
        
    encrypted_key_bytes = bytes.fromhex(encrypted_key_hex)
    try:
        aes_key = rsa_decrypt(encrypted_key_bytes, private_key_path)
        cprint("[SUCCESS] AES Key decrypted successfully.", 'green')
    except Exception as e:
        cprint(f"[FATAL ERROR] Failed to decrypt AES Key: {e}", 'red'); return

    # --- STEP 2A: Recover Nonce and Tag from Frame 0 ---
    cprint("\n[STEP 2A] Recovering Nonce and Tag from Frame 0...", 'cyan')
    NONCE_TAG_BITS = NONCE_TAG_LENGTH * 8 # 256 bits
    frame_0_path = os.path.join(temp_folder, "0.png")

    nonce_tag_binary = extract_data_2bit_lsb(frame_0_path, required_bits=NONCE_TAG_BITS)
    nonce_tag_bytes = binary_to_bytes(nonce_tag_binary)

    tag = nonce_tag_bytes[-16:]
    nonce = nonce_tag_bytes[:-16]

    if len(nonce) != 16 or len(tag) != 16:
        cprint("[FATAL ERROR] Nonce/Tag extraction failed (Wrong size from Frame 0).", 'red'); return
    cprint("[SUCCESS] Nonce and Tag recovered successfully.", 'green')

    # --- STEP 3: Recover and Reconstruct Ciphertext ---
    cprint("\n[STEP 3] Recovering Ciphertext from selected frames...", 'cyan')
    
    full_binary_ciphertext = ""
    bits_extracted = 0
    expected_frames = len(selected_frames)
    avg_bits_per_chunk = math.ceil(total_message_bits / expected_frames)
    
    for i, frame_number in enumerate(selected_frames):
        frame_path = os.path.join(temp_folder, f"{frame_number}.png")
        bits_remaining_overall = total_message_bits - bits_extracted
        bits_to_extract = min(avg_bits_per_chunk, bits_remaining_overall)
        
        if bits_to_extract <= 0: break

        chunk_binary = extract_data_2bit_lsb(frame_path, required_bits=bits_to_extract) 
        full_binary_ciphertext += chunk_binary
        bits_extracted += len(chunk_binary)

        cprint(f"[INFO] Extracted chunk {i+1}/{expected_frames} from frame {frame_number}. Bits taken: {len(chunk_binary)}", 'blue')
        
    if bits_extracted != total_message_bits:
        cprint(f"[FATAL ERROR] Extracted bits ({bits_extracted}) do not match expected total bits ({total_message_bits}).", 'red'); return

    # The full extracted data is the CIPHERTEXT
    ciphertext = binary_to_bytes(full_binary_ciphertext)

    # --- STEP 4: Final Decryption and Header Parsing ---
    cprint("\n[STEP 4] Decrypting Message and Parsing Header...", 'cyan')
    
    decrypted_message = aes_decrypt(aes_key, nonce, tag, ciphertext)
    
    if decrypted_message:
        # --- HEADER PARSING AND OUTPUT LOGIC ---
        if decrypted_message.startswith("TYPE:TEXT|"):
            final_content = decrypted_message.replace("TYPE:TEXT|", "", 1)
            output_type = "RAW TEXT (DISPLAY)"
            
            cprint("\n" + "="*50, 'magenta')
            cprint("               DECODING SUCCESSFUL", 'magenta', attrs=['bold'])
            cprint("="*50, 'magenta')
            cprint(f"Decrypted Data Type: {output_type}", 'green', attrs=['bold'])
            cprint("Extracted Secret Message:", 'green', attrs=['bold'])
            print(final_content)
            cprint("="*50, 'magenta')

        # Inside decode_video_data function:

# ... (Step 4 continued) ...
        elif decrypted_message.startswith("TYPE:FILE|"):
            try:
                _, filename, file_content = decrypted_message.split('|', 2)
                
                output_type = "FILE CONTENT (SAVED TO DISK)"
                
                # --- CRITICAL FIX: Use the Downloads path ---
                output_dir = get_downloads_folder() 
                final_file_path = os.path.join(output_dir, f"DECODED_{filename}")
                
                with open(final_file_path, 'w') as f:
                    f.write(file_content)
                
                # ... (Display success messages) ...
                cprint("\n" + "="*50, 'magenta')
                cprint("               DECODING SUCCESSFUL", 'magenta', attrs=['bold'])
                cprint("="*50, 'magenta')
                cprint(f"Decrypted Data Type: {output_type}", 'green', attrs=['bold'])
                cprint(f"File Name: {filename}", 'green', attrs=['bold'])
                cprint(f"File saved to: {final_file_path}", 'green') # Show new path
                cprint(f"Content Snippet: {file_content[:100]}...", 'yellow')
                cprint("="*50, 'magenta')

            except ValueError:
                cprint("[ERROR] Could not parse file header correctly after decryption.", 'red')
                
        else:
            cprint("\n" + "="*50, 'red')
            cprint("DECODING SUCCESSFUL BUT HEADER CORRUPTED", 'red', attrs=['bold'])
            cprint("="*50, 'red')
            cprint("Raw Decrypted Data:", 'yellow', attrs=['bold'])
            print(decrypted_message[:100] + "...")
            cprint("="*50, 'red')
    
    clean_tmp(temp_folder)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        cprint("Usage: python3 decode.py <stego-video-to-decode-with-extension>", 'red'); sys.exit(1)
        
    video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        cprint(f"[ERROR] Video file not found: {video_file}", 'red'); sys.exit(1)
        
    decode_video_data(video_file)