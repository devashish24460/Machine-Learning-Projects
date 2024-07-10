import os
import shutil
from pypdf import PdfReader
from langdetect import detect, LangDetectException
from datetime import datetime
import re
from pdf_language_detector import PDFLanguageDetector

config_file = 'config.xml'
detector = PDFLanguageDetector(config_file)

def process_pdf(file_path: str, output_folder: str) -> str:
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")

    text = ""
    count = 0
    language_log = []

    timestart = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    language_log.append(f"Processing started at: {timestart}")

    # Regex pattern to identify links and paths
    link_pattern = re.compile(r'http[s]?://|www\.|\.com|\.org|\.net|\.edu|\.gov|\.mil|\.int|file://|/[A-Za-z0-9/._-]+|\\[A-Za-z0-9\\._-]+')

    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except Exception as e:
            language_log.append(f"Failed to extract text from a page: {e}")

    lines = text.splitlines()
    in_link = False

    for line in lines:
        count += 1

        # Print the current line being processed
        print(f"Processing line {count}: {line}")

        if in_link:
            if re.search(r'\s*$', line):
                in_link = False
            else:
                language_log.append(f"Link or path ignored at line {count}")
                continue

        # Check if the line matches the link or path pattern
        if re.search(link_pattern, line):
            in_link = True
            language_log.append(f"Link or path ignored at line {count}")
            continue
        
        try:
            lang = detect(line)
            if lang == "en":
                language_log.append(f"English detected at line {count}")
            else:
                language_log.append(f"Other language ({lang}) detected at line {count}")
        except LangDetectException:
            language_log.append(f"Failed to detect language at line {count} (non-text data)")

    timeend = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    language_log.append(f"Processing ended at: {timeend}")
    language_log.append(f"Total number of lines: {count}")

    # Save the log to a file in the output folder
    output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_output.txt")
    print(f"Output file path: {output_file}")  # Debug print

    try:
        with open(output_file, 'w') as f:
            for log_entry in language_log:
                f.write(log_entry + "\n")
    except Exception as e:
        print(f"Error saving output file: {e}")  # Debug print
        raise  # Raise the exception to propagate it further if needed

    return "\n".join(language_log)

def main():
    # Input and output folder paths from the config file
    input_folder = detector.folders['input_folder']
    output_folder = detector.folders['output_folder']

    # Ensure the input and output folders exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the PDF file path here
    file_path = os.path.join(input_folder, "Processing demo.pdf")

    # Ensure the file path is correctly formatted
    file_path = os.path.abspath(file_path)

    # Check if the file exists and is readable
    if not os.path.isfile(file_path):
        print(f"The file does not exist or is not a valid file: {file_path}")
        return
    if not os.access(file_path, os.R_OK):
        print(f"The file is not readable: {file_path}")
        return

    # Process the PDF file
    try:
        language_result = process_pdf(file_path, output_folder)
        print("Processing completed successfully.")
        print(language_result)
    except Exception as e:
        print(f"Failed to process the PDF file: {e}")

if __name__ == "__main__":
    main()


