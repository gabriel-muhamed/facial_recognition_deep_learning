import os

# Caminho da pasta onde estão suas imagens
folder_path = os.path.join('data', 'jackie_chan')

# Prefixo que você quer dar aos arquivos
prefix = 'jackie_chan'

# Extensões aceitas
valid_extensions = ('.jpg', '.jpeg', '.png')

# Lista arquivos válidos
files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

# Ordena para manter uma ordem consistente
files.sort()

# Renomeia um por um
for idx, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]  # pega a extensão original (.jpg, .png, etc)
    new_name = f"{prefix}_{idx}{ext}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renomeado: {filename} → {new_name}")

print("✅ Renomeação concluída!")
