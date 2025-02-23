import os
import subprocess
from PyPDF2 import PdfMerger
import argparse

def compile_tex_to_pdf(tex_folder, results_folder):
    """
    Compiles all .tex files in a folder into individual PDF files and stores them in a results folder.
    
    Args:
        tex_folder (str): Path to the folder containing .tex files.
        results_folder (str): Path to the folder to store compiled PDF files.
        
    Returns:
        tuple: (list of compiled PDF file paths, bool indicating if any errors occurred)
    """
    pdf_files = []
    has_errors = False  # Flag to track if any compilation errors occurred
    
    # Ensure the input folder exists
    if not os.path.isdir(tex_folder):
        raise FileNotFoundError(f"The folder '{tex_folder}' does not exist.")
    
    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for file_name in os.listdir(tex_folder):
        if file_name.endswith('.tex'):
            tex_file_path = os.path.join(tex_folder, file_name)
            pdf_file_path = os.path.join(results_folder, file_name.replace('.tex', '.pdf'))
            print(f"Compiling: {file_name}")
            
            try:
                # Run pdflatex to compile the .tex file
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_file_path],
                    cwd=results_folder,  # Set results folder as the working directory
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if result.returncode == 0:
                    print(f"Successfully compiled: {pdf_file_path}")
                    pdf_files.append(pdf_file_path)
                else:
                    print(f"Failed to compile: {file_name}")
                    print("Error Output:")
                    print(result.stderr.decode('utf-8'))  # Print the error details
                    has_errors = True  # Set error flag
            except FileNotFoundError:
                raise EnvironmentError("Error: pdflatex is not installed or not in PATH.")
    
    return pdf_files, has_errors

def merge_pdfs(pdf_files, output_pdf):
    """
    Merges multiple PDFs into a single PDF file.
    
    Args:
        pdf_files (list): List of PDF file paths to merge.
        output_pdf (str): Path to the output merged PDF file.
    """
    merger = PdfMerger()
    
    for pdf in pdf_files:
        print(f"Adding to merged PDF: {pdf}")
        merger.append(pdf)
    
    merger.write(output_pdf)
    merger.close()
    print(f"Merged PDF created: {output_pdf}")

def compile_and_merge_tex(tex_folder, output_pdf):
    """
    Compiles all .tex files in a folder into PDFs and merges them into a single PDF.
    
    Args:
        tex_folder (str): Path to the folder containing .tex files.
        output_pdf (str): Path to the output merged PDF file.
    """
    # Generate the results folder name by appending "_results" to the tex_folder name
    results_folder = tex_folder + "_results"
    print(f"Processing .tex files in folder: {tex_folder}")
    print(f"Compiling PDFs into folder: {results_folder}")
    
    # Step 1: Compile .tex files to PDFs
    pdf_files, has_errors = compile_tex_to_pdf(tex_folder, results_folder)
    
    if not pdf_files:
        print("No PDFs were generated. Ensure there are .tex files in the folder.")
        return
    
    # Step 2: Merge all compiled PDFs into a single PDF
    merge_pdfs(pdf_files, output_pdf)
    
    # Step 3: Print final status
    if has_errors:
        print("Finished with errors.")
    else:
        print("Everything worked!")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compile .tex files and merge into a single PDF.")
    parser.add_argument(
        "tex_folder",
        type=str,
        help="Path to the folder containing .tex files"
    )
    parser.add_argument(
        "output_pdf_path",
        type=str,
        help="Path to the output merged PDF file"
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    compile_and_merge_tex(args.tex_folder, args.output_pdf_path)
