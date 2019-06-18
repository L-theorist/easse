from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import time

from imohash import hashfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from ts.evaluation.bleu import get_bleu
from ts.evaluation.sari import get_sari, get_sari_intermediate_scores
from ts.evaluation.readability import get_fre_fkgl, get_dress_fkgl
from ts.exploration import write_comparison_file, df_append_row, compare_multiple_sentences_with_source
from ts.feature_extraction import (get_file_vectorizer, get_levenshtein_distance, count_sentence_splits, is_exact_match,
                                   compression_ratio, new_words_proportion, get_lexical_complexity_score)
from ts.plots import remove_outliers
from ts.preprocess import lowercase_file
from ts.preprocessors import markdown_escape_special_tokens
from ts.quip import QuipAPI
from ts.resources.paths import get_data_filepath, VARIOUS_DIR
from ts.utils import read_file, get_line_lengths, yield_lines, count_lines, yield_lines_in_parallel, mute

'''A simplifier is a method with signature: simplifier(complex_filepath, output_pred_filepath)'''


def get_simplification_scores(complex_filepath, pred_filepath, ref_filepaths=None):
    scores = OrderedDict()
    scores['BLEU'] = (100 * get_bleu(pred_filepath, ref_filepaths)
                      if ref_filepaths is not None else None)
    scores['FKGL'] = get_fre_fkgl(pred_filepath)[1]
    try:
        scores['FKGL'] = get_dress_fkgl(pred_filepath)
    except Exception as e:
        print(e)
    # TODO: Clean up sari evaluation code
    scores['SARI'] = (100 * get_sari(complex_filepath,
                                     pred_filepath,
                                     ref_filepaths,
                                     version='joshuastar',
                                     normalize=True,
                                     lower=False)  # Already lowered
                      if ref_filepaths is not None else None)
    scores['sari_add'], scores['sari_keep'], scores['sari_del'] = get_sari_intermediate_scores(
            complex_filepath,
            pred_filepath,
            ref_filepaths,
            version='joshuastar',
            normalize=True,
            lower=False,  # Already lowered
    ) if ref_filepaths is not None else None
    scores['sari_add'] *= 100
    scores['sari_keep'] *= 100
    scores['sari_del'] *= 100
    vectorizers = [get_levenshtein_distance, count_sentence_splits, is_exact_match, compression_ratio,
                   new_words_proportion]
    for vectorizer in vectorizers:
        scores[vectorizer.__name__] = get_file_vectorizer(vectorizer)(complex_filepath, pred_filepath)
    return {key: round(value, 2) if value is not None else None
            for key, value in scores.items()}


def get_lowercase_simplification_scores(complex_filepath, pred_filepath, ref_filepaths=None):
    complex_filepath = lowercase_file(complex_filepath)
    pred_filepath = lowercase_file(pred_filepath)
    if ref_filepaths is not None:
        ref_filepaths = [lowercase_file(ref_filepath) for ref_filepath in ref_filepaths]
    # TODO: Add lower option
    return get_simplification_scores(complex_filepath, pred_filepath, ref_filepaths)


def evaluate_simplifier_on_file(simplifier, complex_filepath, ref_filepaths=None):
    _, pred_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    return get_lowercase_simplification_scores(complex_filepath, pred_filepath, ref_filepaths)


def evaluate_simplifier_on_turkcorpus(simplifier, phase):
    complex_filepath = get_data_filepath('turkcorpus', phase, 'complex')
    ref_filepaths = [get_data_filepath('turkcorpus', phase, 'simple.turk', i)
                     for i in range(8)]
    return evaluate_simplifier_on_file(simplifier, complex_filepath, ref_filepaths)


def evaluate_simplifier_on_dataset(simplifier, dataset, phase='test'):
    complex_filepath = get_data_filepath(dataset, phase, 'complex')
    ref_filepaths = [get_data_filepath(dataset, phase, 'simple')]
    return evaluate_simplifier_on_file(simplifier, complex_filepath, ref_filepaths)


def evaluate_simplifier_on_wikilarge(simplifier, phase='test'):
    return evaluate_simplifier_on_dataset(simplifier, dataset='wikilarge', phase='test')


def compare_pred_files(source_filepath, pred_filepaths, pred_names, n=float('inf')):
    output = '| Model | Sentence |  \n'  # We need to have a header for the table to be displayed correctly
    i = 0
    for source_line, *pred_lines in yield_lines_in_parallel([source_filepath] + pred_filepaths):
        names = ['Source'] + pred_names
        formatted_sentences = compare_multiple_sentences_with_source(source_line, pred_lines)
        output += tabulate(zip(names, formatted_sentences), tablefmt='pipe') + '  \n'
        i += 1
        if i > n:
            break
    return markdown_escape_special_tokens(output)


def compare_simplifiers_qualitatively(simplifiers):
    complex_filepath = VARIOUS_DIR / 'ts_examples.complex'
    pred_filepaths = []
    names = []
    for simplifier in simplifiers:
        pred_filepath = tempfile.mkstemp()[1]
        simplifier(complex_filepath, pred_filepath)
        pred_filepaths.append(pred_filepath)
        names.append(simplifier.__name__)
    return f'# Qualitative comparison  \n\n  {compare_pred_files(complex_filepath, pred_filepaths, names)}  \n'


def get_sanity_check_text(simplifier, complex_filepath=VARIOUS_DIR / 'ts_examples.complex'):
    '''Displays input sentence, intermediary sentences and pred sentences side by side'''
    temp_dir = Path(tempfile.mkdtemp())

    def mocked_mkstemp():
        '''Mock tempfile.mkstemp() by creating the file in a specific folder with a timestamp in order to track them'''
        path = temp_dir / str(time.time())
        path.touch()
        return 0, path

    original_mkstemp = tempfile.mkstemp
    tempfile.mkstemp = mocked_mkstemp
    timestamped_complex_filepath = tempfile.mkstemp()[1]
    shutil.copyfile(complex_filepath, timestamped_complex_filepath)
    with open(timestamped_complex_filepath, 'a') as f:
        f.write(f'We add this line with a timestamp {time.time()} to change the file hash to prevent memoization .\n')
    pred_filepath = tempfile.mkstemp()[1]
    simplifier(timestamped_complex_filepath, pred_filepath)
    tempfile.mkstemp = original_mkstemp
    # Get temporary files that were created
    created_paths = sorted(temp_dir.glob('*'), key=lambda path: path.stat().st_mtime)
    # Remove duplicate files and empty files
    hashes = []
    paths = []
    n_complex_lines = count_lines(timestamped_complex_filepath)
    for path in [timestamped_complex_filepath] + created_paths + [pred_filepath]:
        if count_lines(path) != n_complex_lines:
            continue
        file_hash = hashfile(path)
        if file_hash in hashes:
            continue
        paths.append(path)
        hashes.append(file_hash)
    output_lines = []
    for lines in yield_lines_in_parallel(paths):
        output_lines += ['\n' + '-' * 80] + lines
    sep = '\n' + '-' * 10 + '\n'
    return f'{sep}## Sanity check  \n' + markdown_escape_special_tokens('  \n'.join(output_lines))


def sanity_check(simplifier, complex_filepath=VARIOUS_DIR / 'ts_examples.complex'):
    print(get_sanity_check_text(simplifier, complex_filepath))


def evaluate_simplifier_qualitatively(simplifier):
    # Cherry picked complex sentences
    complex_filepath = VARIOUS_DIR / 'ts_examples.complex'
    _, pred_filepath = tempfile.mkstemp()
    _, comparison_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    write_comparison_file(complex_filepath, pred_filepath, comparison_filepath)
    output_text = '## Qualitative evaluation  \n'
    sep = '\n' + '-' * 10 + '\n'
    output_text += f'{sep}## Cherry picked complex sentences  \n'
    output_text += read_file(comparison_filepath)
    # Wikilarge predictions sorted with given sort_key
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    _, pred_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    text_key = [
        ('Random Wikilarge predictions',
         lambda c, s: 0),
        ('Wikilarge predictions with the most sentence splits',
         lambda c, s: -count_sentence_splits(c, s)),
        ('Wikilarge predictions with the lowest compression ratio',
         lambda c, s: compression_ratio(c, s)),
        ('Wikilarge predictions with the highest Levenshtein distances',
         lambda c, s: -get_levenshtein_distance(c, s)),
    ]
    for text, sort_key in text_key:
        _, comparison_filepath = tempfile.mkstemp()
        write_comparison_file(complex_filepath, pred_filepath, comparison_filepath, sort_key=sort_key, n_samples=10)
        output_text += f'{sep}## {text}  \n'
        output_text += read_file(comparison_filepath)
    return markdown_escape_special_tokens(output_text)


def evaluate_simplifier_by_sentence_length(simplifier, n_bins=5):
    def get_intervals_from_limits(limits):
        return list(zip(limits[:-1], limits[1:]))

    def get_equally_populated_intervals(filepath, n_bins):
        line_lengths = sorted(get_line_lengths(filepath))
        n_samples_per_bin = int(len(line_lengths) / n_bins)
        limits = [line_lengths[i * n_samples_per_bin] for i in range(n_bins)] + [line_lengths[-1] + 1]
        return get_intervals_from_limits(limits)

    def split_lines_by_lengths(filepath, intervals):
        bins = [[] for _ in range(len(intervals))]
        for line_idx, line in enumerate(yield_lines(filepath)):
            line_length = len(line)
            for interval_idx, (interval_start, interval_end) in enumerate(intervals):
                if interval_start <= line_length and line_length < interval_end:
                    bins[interval_idx].append(line_idx)
                    break
        assert sum([len(b) for b in bins]) == count_lines(filepath)
        return bins

    def select_lines(input_filepath, output_filepath, line_indexes):
        line_indexes = set(line_indexes)
        with open(output_filepath, 'w') as f:
            for line_idx, line in enumerate(yield_lines(input_filepath)):
                if line_idx in line_indexes:
                    f.write(line + '\n')

    def split_file_by_bins(input_filepath, bins):
        splitted_filepaths = [tempfile.mkstemp()[1] for _ in range(len(bins))]
        for splitted_filepath, line_indexes in zip(splitted_filepaths, bins):
            select_lines(input_filepath, splitted_filepath, line_indexes)
        return splitted_filepaths

    # Run predicition
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    ref_filepath = get_data_filepath('wikilarge', 'test', 'simple')
    _, pred_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    # Get line length bins
    intervals = get_equally_populated_intervals(complex_filepath, n_bins)
    bins = split_lines_by_lengths(complex_filepath, intervals)
    # Split files by bins
    splitted_complex_filepaths = split_file_by_bins(complex_filepath, bins)
    splitted_ref_filepaths = split_file_by_bins(ref_filepath, bins)
    splitted_pred_filepaths = split_file_by_bins(pred_filepath, bins)
    df_bins = pd.DataFrame()
    # Get scores for each bin
    for i in range(len(intervals)):
        interval = intervals[i]
        splitted_complex_filepath = splitted_complex_filepaths[i]
        splitted_pred_filepath = splitted_pred_filepaths[i]
        splitted_ref_filepath = splitted_ref_filepaths[i]
        scores = get_simplification_scores(splitted_complex_filepath,
                                           splitted_pred_filepath,
                                           [splitted_ref_filepath])
        row_name = f'{simplifier.__name__}_{interval[0]}_{interval[1]}'
        df_bins = df_append_row(df_bins, scores, row_name)
    return df_bins


def get_markdown_scores(simplifier):
    '''Return a markdown formatted string of turkcorpus and wikilarge scores'''
    df_scores = pd.DataFrame()
    for phase in ['valid', 'test']:
        turkcorpus_scores = evaluate_simplifier_on_turkcorpus(simplifier, phase=phase)
        df_scores = df_append_row(df_scores, turkcorpus_scores, f'Turkcorpus ({phase})')
        # wikilarge_scores = evaluate_simplifier_on_wikilarge(simplifier, phase=phase)
        # df_scores = df_append_row(df_scores, wikilarge_scores, f'Wikilarge ({phase})')
    scores_table = tabulate(df_scores, headers='keys', tablefmt='pipe')
    return f'## Scores and metrics  \n{scores_table}  \n\n'


def get_markdown_scores_by_sentence_length(simplifier):
    '''Return a markdown formatted table string of scores broken down by sentence length'''
    df_bins = evaluate_simplifier_by_sentence_length(simplifier)
    scores_table = tabulate(df_bins, headers='keys', tablefmt='pipe')
    return f'## Wikilarge scores broken down by sentence length:  \n{scores_table}  \n\n'


def get_quip_image_html(image_path, thread_id, api):
    '''Upload a blob to Quip and return the HTML to be included in the document'''
    with open(image_path, 'rb') as f:
        response = api.client.put_blob(thread_id, f)
    blob_url = response['url']
    return f'<img src="{blob_url}"</img>\n'


def get_compression_ratio_html_plot(simplifier, thread_id, api):
    def get_compression_ratios(simplifier):
        complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
        pred_filepath = tempfile.mkstemp()[1]
        simplifier(complex_filepath, pred_filepath)
        compression_ratios = []
        for complex_line, pred_line in yield_lines_in_parallel([complex_filepath, pred_filepath]):
            compression_ratios.append(compression_ratio(complex_line, pred_line))
        return compression_ratios

    plot_path = Path(tempfile.mkdtemp()) / 'tmp.png'
    compression_ratios = get_compression_ratios(simplifier)
    sns.set_style('darkgrid')
    plt.hist(compression_ratios, bins=30, range=[0, 2])
    plt.margins(x=0)
    plt.title(f'Compression ratio distribution\n(Wikilarge test, {len(compression_ratios)} samples)')
    plt.xlabel('Compression ratio')
    plt.ylabel('Frequency')
    plt.savefig(plot_path)
    plt.gcf().clear()
    return get_quip_image_html(plot_path, thread_id, api)


def get_length_html_plot(simplifier, thread_id, api):
    plot_path = Path(tempfile.mkdtemp()) / 'tmp.png'
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    pred_filepath = tempfile.mkstemp()[1]
    simplifier(complex_filepath, pred_filepath)
    X, Y = zip(*[(len(complex_line), len(pred_line))
                 for complex_line, pred_line
                 in yield_lines_in_parallel([complex_filepath, pred_filepath])])
    # Remove outliers
    X, Y = remove_outliers(X, Y)
    plt.scatter(X, Y, s=3)
    plt.xlim(0, max(np.max(X), np.max(Y)))
    plt.ylim(0, max(np.max(X), np.max(Y)))
    plt.title('Lengths of prediction w.r.t. source')
    plt.xlabel('Source length (characters)')
    plt.ylabel('Prediction length (characters)')
    plt.savefig(plot_path)
    plt.gcf().clear()
    return get_quip_image_html(plot_path, thread_id, api)


def get_lexical_complexity_html_plot(simplifier, thread_id, api):
    plot_path = Path(tempfile.mkdtemp()) / 'tmp.png'
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    pred_filepath = tempfile.mkstemp()[1]
    simplifier(complex_filepath, pred_filepath)
    X, Y = zip(*[(get_lexical_complexity_score(complex_line), get_lexical_complexity_score(pred_line))
                 for complex_line, pred_line
                 in yield_lines_in_parallel([complex_filepath, pred_filepath])])
    X, Y = remove_outliers(X, Y)
    plt.scatter(X, Y, s=3)
    plt.xlim(0, max(np.max(X), np.max(Y)))
    plt.ylim(0, max(np.max(X), np.max(Y)))
    plt.title('Lexical complexity score of prediction w.r.t. source')
    plt.xlabel('Source lexical complexity score')
    plt.ylabel('Prediction lexical complexity score')
    plt.savefig(plot_path)
    plt.gcf().clear()
    return get_quip_image_html(plot_path, thread_id, api)


def get_markdown_report(simplifier):
    text = f'# {simplifier.__name__}  \n'
    text += get_markdown_scores(simplifier)
    text += get_markdown_scores_by_sentence_length(simplifier)
    text += evaluate_simplifier_qualitatively(simplifier)
    text += get_sanity_check_text(simplifier)
    return text


def get_quip_report(api, simplifier):
    text = f'# {simplifier.__name__}  \n'
    text += get_markdown_scores(simplifier)
    text += get_compression_ratio_html_plot(simplifier, api.get_document_id(simplifier.__name__), api)
    text += get_length_html_plot(simplifier, api.get_document_id(simplifier.__name__), api)
    text += get_lexical_complexity_html_plot(simplifier, api.get_document_id(simplifier.__name__), api)
    text += get_markdown_scores_by_sentence_length(simplifier)
    text += evaluate_simplifier_qualitatively(simplifier)
    text += get_sanity_check_text(simplifier)
    return text


def get_wikilarge_sota():
    columns = ['BLEU', 'FKGL', 'SARI']
    df_wikilarge = pd.DataFrame(columns=columns)
    df_wikilarge.loc['PBMT-R', columns] = [81.11, 8.33, 38.56]
    df_wikilarge.loc['Hybrid', columns] = [48.97, 4.56, 31.40]
    df_wikilarge.loc['SBMT-SARI', columns] = [73.08, 7.29, 39.96]
    df_wikilarge.loc['EncDecA', columns] = [88.85, 8.41, 35.66]
    df_wikilarge.loc['DRESS-LS', columns] = [80.12, 6.62, 37.27]
    return df_wikilarge


class ExperimentQuipAPI(QuipAPI):
    '''API to manipulate a single experiment in quip'''

    def __init__(self, folder_id, token):
        super().__init__(token)
        self.folder_id = folder_id
        self.index_folder(folder_id)
        self.scores_table_id = self.get_element_id_in_folder('scores_table', folder_id)
        self.detailed_results_folder_id = self.get_element_id_in_folder('detailed_results', folder_id)
        self.index_folder(self.detailed_results_folder_id)

    def get_document_id(self, title):
        return self.get_document_id_in_folder(title, self.detailed_results_folder_id)

    def get_content_hash(self, thread_id):
        match = re.search(r'content_hash=([a-f0-9]+)', self._indexed_responses[thread_id]['html'])
        return match.groups()[0] if match is not None else ''

    def upload_results(self, title, content):
        thread_id = self.get_document_id(title)
        # Check if the content is different using its hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if self.get_content_hash(thread_id) == content_hash:
            # No modification
            return
        content = f'content_hash={content_hash}  \n' + content
        # Separate old content from the content we are uploading
        sep = '\n' + '-'*10 + '\n'
        content += '  \n'*20 + sep + sep + '# The following is outdated  \n' + sep
        print(f'Editing document "{title}"')
        response = self.client.edit_document(thread_id,
                                             content=content,
                                             operation=self.client.PREPEND,
                                             section_id=None,
                                             format='markdown')
        self.read_and_save_response(response)

    def upload_scores_in_table(self, model_name, updates, scores_table_id):
        # Keep only valid updates (i.e. remove those that are already present)
        current_spreadsheet = self.client.get_first_spreadsheet(document_html=self.get_document_html(scores_table_id))
        updates = self.get_valid_spreadsheet_updates(current_spreadsheet, 'Model', model_name, updates)
        if len(updates) == 0:
            return
        print(f'Updating Quip results for model "{model_name}" with {updates}')
        response = self.client.update_spreadsheet_row(self.scores_table_id, 'Model', model_name, updates)
        self.read_and_save_response(response)

    def evaluate_and_upload_simplifier(self, simplifier):
        # Upload detailed results in dedicated document
        with mute():
            turkcorpus_scores = evaluate_simplifier_on_turkcorpus(simplifier, phase='valid')
            wikilarge_scores = evaluate_simplifier_on_wikilarge(simplifier, phase='valid')
            content = get_quip_report(self, simplifier)
        self.upload_results(title=simplifier.__name__, content=content)
        # Upload scores in experiment table
        turkcorpus_scores = {f'{key} (turkcorpus)': turkcorpus_scores[key] for key in ['BLEU', 'FKGL', 'SARI']}
        updates = {k: str(v)
                   for k, v in list(turkcorpus_scores.items()) + list(wikilarge_scores.items())
                   if v is not None}
        # Add the link to the dedicated results document
        updates['Detailed results'] = self.get_document_link(self.get_document_id(simplifier.__name__))
        self.upload_scores_in_table(simplifier.__name__, updates, self.scores_table_id)
