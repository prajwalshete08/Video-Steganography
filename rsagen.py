# key_generator.py

from Crypto.PublicKey import RSA
from termcolor import cprint
import os

def generate_rsa_keys(key_size=2048, filename_prefix="new_user"):
    """
    Generates a private/public RSA key pair and saves them to a 'keys' subdirectory.
    
    Args:
        key_size (int): The size of the RSA modulus in bits (e.g., 2048).
        filename_prefix (str): Prefix for the output files (e.g., 'new_user').
    """
    
    # --- 1. Define and Create Keys Directory ---
    KEYS_DIR = "keys"
    try:
        # Create the 'keys' directory if it doesn't exist
        os.makedirs(KEYS_DIR, exist_ok=True)
        cprint(f"[INFO] Keys directory '{KEYS_DIR}' ensured.", 'blue')
    except Exception as e:
        cprint(f"[FATAL ERROR] Could not create directory '{KEYS_DIR}': {e}", 'red')
        return

    cprint(f"\n--- Generating {key_size}-bit RSA Key Pair ---", 'cyan')
    
    # 2. Generate the private key
    key = RSA.generate(key_size)
    
    # 3. Extract the public key
    public_key = key.publickey()
    
    # Define file names inside the KEYS_DIR
    private_filename = f"{filename_prefix}_private_{key_size}.pem"
    public_filename = f"{filename_prefix}_public_{key_size}.pem"
    
    private_file_path = os.path.join(KEYS_DIR, private_filename)
    public_file_path = os.path.join(KEYS_DIR, public_filename)
    
    # 4. Save Private Key (PKCS#8 format is common and secure)
    try:
        with open(private_file_path, 'wb') as f:
            # Export the private key in PEM format
            f.write(key.export_key('PEM'))
        cprint(f"[SUCCESS] Private key saved to: {os.path.abspath(private_file_path)}", 'green')
    except Exception as e:
        cprint(f"[ERROR] Could not save private key: {e}", 'red')
        return

    # 5. Save Public Key
    try:
        with open(public_file_path, 'wb') as f:
            # Export the public key in PEM format
            f.write(public_key.export_key('PEM'))
        cprint(f"[SUCCESS] Public key saved to: {os.path.abspath(public_file_path)}", 'green')
    except Exception as e:
        cprint(f"[ERROR] Could not save public key: {e}", 'red')


if __name__ == "__main__":
    # Keys will be saved in the new 'keys' directory:
    # keys/stego_rsa_private_2048.pem
    # keys/stego_rsa_public_2048.pem
    generate_rsa_keys(key_size=2048, filename_prefix="stego_rsa")