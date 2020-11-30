""" accepts data of format Title <EOT> Story"""
import argparse
import json
import logging
# TO DO enable it to take stories and titles or just stories
from pathlib import Path
import re
import sys
from typing import Dict, List

from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp_models.coref import CorefPredictor
from allennlp_models.structured_prediction import SemanticRoleLabelerPredictor
from attr import dataclass
import spacy
from tqdm import tqdm

EOS: str = "</s>"


@dataclass
class Sentence(object):
    """Models a sentence"""
    content: str
    begin: int
    end: int


class Story(object):
    """docstring for Story"""

    def __init__(self, tokens: List[str]):
        super(Story, self).__init__()
        self.tokens = tokens

    def join_sentence(self):
        """
        After using AllenNLP coref parser, use the word_tokenzied list from a whole story to join
        a sent_tokenized list. Sep flag is </s>.
        """
        idx = 0
        length = len(self.tokens)
        pre_idx = 0
        curent_string = ''
        sentences = []
        while idx < len(self.tokens):
            if self.tokens[idx] == EOS and idx + 1 < length:
                # if self.char_list[idx] == '<' and idx + 2 < length and self.char_list[idx + 1] ==
                # '/s' and self.char_list[idx + 2] == '>':
                sentence = Sentence(curent_string[:len(curent_string) - 1], pre_idx, idx)
                sentences.append(sentence)
                curent_string = ''
                # pre_idx = idx = idx + 3
                pre_idx = idx = idx + 1
            else:
                curent_string = curent_string + self.tokens[idx] + " "
                idx += 1
        sentence = Sentence(curent_string[:len(curent_string) - 1], pre_idx, idx)
        sentences.append(sentence)
        return sentences


def load_stories(infile: Path) -> List[str]:
    documents = []
    with infile.open(mode="r") as inf:
        for line in inf:
            text = " ".join(line.strip().split("\t"))
            documents.append(text)
    return documents


def coref_resolution(text, CorefPredictor):
    """
    using Allennlp pretranied model to do coreference resolution
    :param text: a story
    :param coref_model: pretrained model weight, you should define its path in hyperpramenter
    :param cuda_device: if it >=0, it will load archival model on GPU otherwise CPU
    :return:  first return is a list of word_tokennize list of one story,
            second returen is a three layers list, [[[1,1],[3,5]],[6,6],[8,11]], same entity's
            index will be clusted together
    """
    result = CorefPredictor.predict_tokenized(text)
    return result.get("document"), result.get("clusters")


def predict_srl(text, srl_predictor, batch_size):
    """
    :param text: a string of  story
    :param srl_predictor:
    :param batch_size: Size of batches.
    :param cuda_device: if it >=0, it will load archival model on GPU otherwise CPU
    :return: all predictions after srl
    """

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = srl_predictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = srl_predictor.predict_batch_json(batch_data)
        return results

    batch_data = []
    all_predictions = []
    for line in text.split(EOS):
        if not line.isspace():
            line = {"sentence": line.strip()}
            line = json.dumps(line)
            json_data = srl_predictor.load_line(line)
            batch_data.append(json_data)
            if len(batch_data) == batch_size:
                predictions = _run_predictor(batch_data)
                all_predictions.append(predictions)
                batch_data = []
    if batch_data:
        predictions = _run_predictor(batch_data)
        all_predictions.append(predictions)

    all_description = []
    for batch in all_predictions:
        for sentence in batch:
            verbs = sentence.get("verbs")
            description = []
            for verb in verbs:
                description.append(verb.get("description"))
            all_description.append(description)
    return all_description


# range_list = parse(dec, doc, doc_current_index)
def extract_storyline(doc, clusters, predictor_srl, batch_size):
    """
    After getting all srl anf coref clusters, we need check if one ARG is in clusters, if so we 
    need to change it to "ent{}"
    :param doc:
    :param clusters:
    :param srl_model:
    :param batch_size:
    :param cuda_device:
    :return:
    """
    document = Story(doc)
    sentences = document.join_sentence()
    text = " ".join(document.tokens)
    all_descriptions = predict_srl(text, predictor_srl, batch_size)
    storyline = []
    if len(sentences) != len(all_descriptions):
        assert ("SRL WRONG, the length of sentence is not equal to length of descriptions")
    for s in sentences:
        descriptions = all_descriptions[sentences.index(s)]
        for description in descriptions:
            sentence_description = {}
            items = re.findall(r"\[(.+?)\]+?", description)  # only context
            for item in items:
                tag = item.split(": ")[0]
                if tag == "V":
                    sentence_description["<V>"] = item.split(': ')[1]
                elif tag in ["ARG0", "ARG1", "ARG2"]:
                    new_argument = replace_ent(item, s, doc, clusters)
                    for i in range(0, 3):
                        if tag == "ARG{}".format(i):
                            sentence_description["<A{}>".format(i)] = new_argument
            sentence_description = compress(sentence_description)
            if len(sentence_description) > 0:
                storyline.append(sentence_description)
                storyline.append("#")
        storyline.append(EOS)
    return storyline, all_descriptions


def intersection(list1, list2):
    """
    helper function to find wheter srl argument index overlap with coref_resolution clusters list
    :param list1:
    :param list2:
    :return: the intersection part of two list
    """
    l = max(list1[0], list2[0])
    r = min(list1[1], list2[1])
    if l > r:
        return []
    return [l, r]


def replace_ent(argument, sentence, doc, clusters):
    """
    comparing the srl results and coreference resolution results,
    and change "ARG{}" to "ent{}" if in clusters
    """
    sub_sentence = argument.split(': ')[1]
    sub_sentence_words = sub_sentence.split(' ')
    new_argument = ''
    begin = end = -1
    for i in range(sentence.begin, sentence.end - len(sub_sentence_words)):
        is_match = True
        for j in range(len(sub_sentence_words)):
            if sub_sentence_words[j] != doc[i + j]:
                is_match = False
                break
        if is_match:
            begin = i
            end = i + len(sub_sentence_words)
            break
    for ent_idx in range(len(clusters)):
        for ent_range in clusters[ent_idx]:
            intersection_range = intersection(ent_range, [begin, end])
            if len(intersection_range) > 0:
                for replace_idx in range(0, min(len(sub_sentence_words),
                                                intersection_range[1] - intersection_range[0] + 1)):
                    sub_sentence_words[replace_idx] = "ent {}".format(ent_idx)
    for i in range(len(sub_sentence_words)):
        if i == 0 or sub_sentence_words[i - 1] != sub_sentence_words[i]:
            new_argument += sub_sentence_words[i]
        else:
            continue
        if i != len(sub_sentence_words) - 1:
            new_argument += ' '
    return new_argument


def compress(sentence_description: Dict[str, str]) -> Dict:
    """
        Compress long and messy SRL output to more abstract. This function is mostly equivalent to
        the compression described in Appendix A.5.
    """
    new_dic = sentence_description
    # Rule 1: Delete lines which only have V, since SRL aim is to learn info like "who does what",
    # or "who does what to whom". If the sentence only has a verb prediction, it is useless.
    if "<A0>" not in sentence_description and "<A1>" not in sentence_description and "<A2>" not \
            in sentence_description:
        new_dic = {}
        return new_dic
    # Rule 2:  Delete lines whose Verb is "be" or modal verb.
    # Appendix A.5 in the paper.
    if sentence_description.get("<V>") in ["is", "was", "were", "are", "be", "\'s", "\'re", "\'ll",
                                           "can", "could", "must", "may", "have to", "has to",
                                           "had to", "will", "would", "has", "have", "had", "do",
                                           "does", "did"]:
        new_dic = {}
        return new_dic
    # Rule 3: Delete lines whose AGR length exceed 5.
    # TODO: This is inconsistent with A.5 which states that arguments past 0, 1, 2 should be
    #  removed, not the entire sentence.
    for i in range(0, 3):
        if f"<A{i}>" in sentence_description and len(
                sentence_description.get(f"<A{i}>").split(" ")) > 5:
            new_dic = {}
            return new_dic
    return new_dic


def label_story(doc, cluster):
    for i, item in enumerate(cluster):
        for ent in item:
            beg = ent[0]
            end = ent[1]
            # TODO change this logic
            doc[beg] = "<ent> {0} {1}".format(i, doc[beg])
            doc[end] = "{0} </ent> {1}".format(doc[end], i)
    labeled_story = " ".join(doc)
    return labeled_story


def spacy_word_token(text, nlp):
    doc = nlp(text)
    token_list = [t.text for t in doc]
    return token_list


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


def predict_coref(documents, coref_model: str, cuda: int, nlp) -> List[Dict]:
    """
    Resolves coreference on the given list of documents.
    """
    all_json = []
    corefpredictor = CorefPredictor.from_archive(load_archive(coref_model, cuda))
    for text in tqdm(documents):
        elements = text.split(" <EOT> ")
        prompt = elements[0]
        story = elements[1]
        try:
            story = spacy_word_token(story, nlp)
        except:
            logging.info("Story is empty, out of range")
        try:
            doc, clusters = coref_resolution(story, corefpredictor)
            all_json.append({"doc": doc, "clusters": clusters, "prompt": prompt})

        except RuntimeError:
            logging.info("Runtime Error")
    return all_json


def handle_srl(srl_model: str, cuda: int, docs_clusters, batch: int):
    predictor_srl = SemanticRoleLabelerPredictor.from_archive(load_archive(srl_model, cuda))
    all_storyline = []
    all_prediction = []
    all_labeled_stories = []
    all_title = []
    for text in tqdm(docs_clusters):
        doc, clusters, prompt = text["doc"], text["clusters"], text["prompt"]
        all_title.append(prompt)
        text_info = {}
        storyline, srl = extract_storyline(doc, clusters, predictor_srl, batch)
        labeled_story = label_story(doc, clusters)

        if len(storyline) > 0:
            all_storyline.append(storyline)
            text_info["doc"] = doc
            text_info["clusters"] = clusters
            text_info["srl"] = srl
            text_info["prompt"] = prompt
            all_prediction.append(text_info)
            all_labeled_stories.append(labeled_story)
    return all_storyline, all_prediction, all_labeled_stories, all_title


def main():
    """
    Main method
    """

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = ParserWithUsage()
    parser.description = "Runner description"
    parser.add_argument("--input_file", help="Path to file containing stories", type=Path,
                        required=True)
    parser.add_argument('--output_file', type=Path, help='path to output file', required=True)
    parser.add_argument('--coref_model', type=str,
                        help='Path to pretrained model weight for co-refrence resolution',
                        required=True)
    parser.add_argument('--srl_model', type=str,
                        help='path to pretrained mode weight for semantic role labeler',
                        required=True)
    parser.add_argument('--batch', type=int, default=1, help='The batch size to use for processing')
    parser.add_argument('--cuda', type=int, default=-1, help='ID of GPU to use (if any)')
    parser.add_argument('--save_coref_srl', type=Path,
                        help='Path for saving coref clusters and doc and srl for reuse',
                        required=True)
    parser.add_argument('--label_story', type=Path,
                        help='Path for saving the stories after add ent label', required=True)
    parser.add_argument('--title', type=Path, help='Path for saving the titles', required=True)
    # parser.add_argument('--reusem',  action='store true', help='reusem the coref, srl prediction')
    args = parser.parse_args()
    output_file = args.output_file
    out_title = args.title
    out_label_story = args.label_story
    out_coref_srl = args.save_coref_srl

    # Check GPU
    check_for_gpu(args.cuda)

    # Hardcoded list of special characters not to touch
    special_chars = {'<EOL>', '<EOT>', '<eos>', EOS, '#', '<P>', "``", "\"", '[UNK]'}

    # load spacy tokenizer
    spacy_model = 'en_core_web_sm'
    # TODO add spacy check here to autodownload rather than error
    nlp = spacy.load(spacy_model)
    # Need to special case all special chars for tokenization
    for key in special_chars:
        nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])

    documents = load_stories(args.input_file)
    logging.info(f"Loaded {len(documents)} stories!")

    docs_clusters = predict_coref(documents, args.coref_model, args.cuda, nlp)

    # Now to SRL
    storylines, predictions, labeled_stories, titles = handle_srl(args.srl_model,
                                                                  args.cuda,
                                                                  docs_clusters,
                                                                  args.batch
                                                                  )

    if len(storylines) > 0:
        logging.info(f"Save {len(storylines)} storylines!")
        with output_file.open(mode="w", encoding='utf8') as fout:
            json.dump(storylines, fout, ensure_ascii=False, indent=4)

        logging.info(f"Save {len(titles)} valid titles!")
        with out_title.open(mode="w", encoding='utf8') as fout:
            json.dump(titles, fout, ensure_ascii=False, indent=4)

        logging.info(f"Save {len(predictions)} valid and labeled stories!")
        with out_label_story.open(mode="w", encoding='utf8') as fout:
            json.dump(labeled_stories, fout, ensure_ascii=False, indent=4)

        logging.info(f"Save {len(predictions)} coref and srl predictions!")
        with out_coref_srl.open(mode="w", encoding='utf8') as fout:
            json.dump(predictions, fout, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
