import os
import subprocess
import re
from PyPDF2 import PdfMerger
import argparse


def fix_png_paths(tex_file_path):
    r"""
    Reads a .tex file, replaces PNG file references with absolute paths,
    and writes the modified content to a temporary file.

    For every occurrence of a PNG file (e.g. in commands like:
    \includegraphics{image.png}), it replaces the file name with the absolute
    directory (with forward slashes) of the current .tex file plus a '/'
    and the base name of the png file.

    Args:
        tex_file_path (str): Path to the original .tex file.

    Returns:
        str: Path to the temporary modified .tex file.
    """
    with open(tex_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Compute the absolute directory of the .tex file and force forward slashes.
    abs_dir = os.path.abspath(os.path.dirname(tex_file_path)).replace(os.sep, "/")

    # Pattern to match any string ending with '.png' inside curly braces.
    # For example, matches "image.png" in \includegraphics{image.png}
    pattern = re.compile(r"(?<=\{)([^}]*?\.png)(?=\})")

    def replace(match):
        original_path = match.group(0).strip()
        base = os.path.basename(original_path)
        new_path = abs_dir + "/" + base
        return new_path

    new_content = re.sub(pattern, replace, content)
    # Write the modified content to a temporary file.
    temp_file_path = tex_file_path.replace(".tex", "_temp.tex")
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return temp_file_path


def compile_tex_to_pdf(tex_folder, results_folder):
    r"""
    Compiles all .tex files in a folder into individual PDF files and stores
    them in a results folder.

    For each .tex file, the function:
      1. Updates the file’s PNG references (using fix_png_paths).
      2. Compiles the temporary modified file with pdflatex. (We do not use -jobname;
         instead, we let pdflatex use the temporary file’s name and then rename the
         PDF afterwards.)
      3. Deletes the temporary file after compilation.
      4. Renames the generated PDF from <name>_temp.pdf to <name>.pdf.

    Args:
        tex_folder (str): Path to the folder containing .tex files.
        results_folder (str): Path to the folder to store compiled PDF files.

    Returns:
        tuple: (list of compiled PDF file paths, bool indicating if any errors occurred)
    """
    pdf_files = []
    has_errors = False

    # Ensure the input folder exists.
    if not os.path.isdir(tex_folder):
        raise FileNotFoundError(f"The folder '{tex_folder}' does not exist.")

    # Ensure the results folder exists.
    os.makedirs(results_folder, exist_ok=True)
    abs_results_folder = os.path.abspath(results_folder)
    abs_tex_folder = os.path.abspath(tex_folder)

    for file_name in os.listdir(tex_folder):
        if file_name.endswith(".tex"):
            orig_tex_path = os.path.join(tex_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            # Our target PDF name.
            pdf_file_path = os.path.join(abs_results_folder, base_name + ".pdf")
            print(f"Compiling: {file_name}")

            try:
                # Create a temporary modified file with absolute paths for PNGs.
                temp_tex_file = fix_png_paths(orig_tex_path)
                temp_file_basename = os.path.basename(temp_tex_file)
                
                # Run pdflatex on the temporary file.
                # Note: We no longer use -jobname so that pdflatex uses the
                # temporary file's basename. We then rename the output PDF.
                cmd = [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    abs_results_folder,
                    temp_file_basename,
                ]
                result = subprocess.run(
                    cmd,
                    cwd=abs_tex_folder,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Remove the temporary .tex file.
                os.remove(temp_tex_file)

                if result.returncode == 0:
                    # The PDF will be named as <basename>_temp.pdf.
                    temp_pdf = os.path.join(
                        abs_results_folder, temp_file_basename.replace(".tex", ".pdf")
                    )
                    if os.path.exists(temp_pdf):
                        # Rename it to the desired PDF name.
                        os.rename(temp_pdf, pdf_file_path)
                        print(f"Successfully compiled: {pdf_file_path}")
                        pdf_files.append(pdf_file_path)
                    else:
                        print(
                            f"Compilation succeeded but PDF not found for: {file_name}"
                        )
                        has_errors = True
                else:
                    print(f"Failed to compile: {file_name}")
                    out_stdout = result.stdout.decode("utf-8").strip()
                    out_stderr = result.stderr.decode("utf-8").strip()
                    if out_stdout:
                        print("Output:")
                        print(out_stdout)
                    if out_stderr:
                        print("Error Output:")
                        print(out_stderr)
                    has_errors = True
            except FileNotFoundError:
                raise EnvironmentError(
                    "Error: pdflatex is not installed or not in PATH."
                )

    return pdf_files, has_errors


def merge_pdfs(pdf_files, output_pdf):
    r"""
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
    r"""
    Compiles all .tex files in a folder into PDFs and then merges them into a
    single PDF.

    It creates a results folder (by appending '_results' to the tex_folder name)
    to store individual PDFs before merging.

    Args:
        tex_folder (str): Path to the folder containing .tex files.
        output_pdf (str): Path to the output merged PDF file.
    """
    results_folder = tex_folder + "_results"
    print(f"Processing .tex files in folder: {tex_folder}")
    print(f"Compiling PDFs into folder: {results_folder}")

    # Step 1: Compile .tex files to PDFs.
    pdf_files, has_errors = compile_tex_to_pdf(tex_folder, results_folder)

    if not pdf_files:
        print("No PDFs were generated. Ensure there are .tex files in the folder.")
        return

    # Step 2: Merge all compiled PDFs into a single PDF.
    merge_pdfs(pdf_files, output_pdf)

    # Step 3: Final status.
    if has_errors:
        print("Finished with errors.")
    else:
        print("Everything worked!")


if __name__ == "__main__":
    # Set up argument parsing.
    parser = argparse.ArgumentParser(
        description="Compile .tex files and merge into a single PDF."
    )
    parser.add_argument(
        "tex_folder", type=str, help="Path to the folder containing .tex files"
    )
    parser.add_argument(
        "output_pdf_path", type=str, help="Path to the output merged PDF file"
    )

    args = parser.parse_args()
    compile_and_merge_tex(args.tex_folder, args.output_pdf_path)
