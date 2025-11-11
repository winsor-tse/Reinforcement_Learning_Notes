import os

def srt_to_text(srt_path, txt_path):
    with open(srt_path, 'r', encoding='utf-8') as srt_file:
        lines = srt_file.readlines()

    subtitle_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        subtitle_lines.append(line)

    full_text = ' '.join(subtitle_lines)
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(full_text)

def convert_all_srts_in_dir(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.srt'):
            srt_path = os.path.join(directory, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(directory, txt_filename)
            srt_to_text(srt_path, txt_path)
            print(f"Converted: {filename} → {txt_filename}")

# Example usage
convert_all_srts_in_dir("Generalization")

import os

def combine_txt_files(directory, output_file="combined.txt"):
    with open(os.path.join(directory, output_file), 'w', encoding='utf-8') as outfile:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.txt') and filename != output_file:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    if content:
                        outfile.write(content + '\n\n')  # Add double newline between files
                print(f"Added: {filename}")
    print(f"All files combined into {output_file}")

# Example usage
combine_txt_files("Generalization")
