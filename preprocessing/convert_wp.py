import argparse
import logging
from pathlib import Path
import sys
from typing import List


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message) -> None:
        """
        Prints error message and help.
        :param message: error message to print
        """
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def load_line_documents(file: Path) -> List[str]:
    with file.open(mode="r") as infile:
        docs = [line for line in infile]
    return docs


def clean_prompts(prompts: List[str]) -> List[str]:
    import re
    new_prompts = [re.sub("\[.*\]\s", "", s.replace("\n", "")) for s in prompts]
    return new_prompts


def process_newline(stories: List[str]) -> List[str]:
    new_stories = [s.replace("<newline>", "<P>") for s in stories]
    return new_stories


def split_sentences(stories: List[str]) -> List[str]:
    from spacy.lang.en import English
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))  # updated
    new_stories = []
    for s in stories:
        doc = nlp(s)
        sentences = [sent.string.strip() for sent in doc.sents]
        new_story = " </s> ".join(sentences)
        new_stories.append(new_story)
    return new_stories


def main():
    """
    Main method
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Converts WritingPrompts to Title <EOT> Story sent 1 <\s> Story sent 2"
    parser.add_argument("--input-prompts", help="Path to prompts file", required=True, type=Path)
    parser.add_argument("--input-stories", help="Path to file with stories", required=True,
                        type=Path)
    parser.add_argument("--output", help="Output directory", required=True, type=Path)

    args = parser.parse_args()
    logging.info("STARTED")
    path_stories = args.input_stories
    stories = load_line_documents(path_stories)
    stories = process_newline(stories)
    stories = split_sentences(stories)
    path_prompts = args.input_prompts
    prompts = load_line_documents(path_prompts)
    prompts = clean_prompts(prompts)

    path_output = args.output

    path_output_stories = path_output / f"{path_stories.stem}.stories"
    with path_output_stories.open(mode="w") as outfile:
        for s in stories:
            outfile.write(s)
            outfile.write("\n")

    path_output_prompts = path_output / f"{args.input_prompts.stem}.titles"
    with path_output_prompts.open(mode="w") as outfile:
        for p in prompts:
            outfile.write(p)
            outfile.write("\n")


    path_output_merged = path_output / f"{args.input_prompts.stem}.clean.txt"
    with path_output_merged.open(mode="w") as outfile:
        for p, s in zip(prompts, stories):
            outfile.write(p)
            outfile.write(" <EOT> ")
            outfile.write(s)
            outfile.write("\n")

    logging.info("DONE")


if __name__ == "__main__":
    main()
