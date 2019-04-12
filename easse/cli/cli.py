import click
import sacrebleu

import easse.cli.utils as cli_utils
import easse.sari.sari_score as sari
import easse.samsa.samsa_score as samsa
import easse.annotation.word_level as annotation

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    pass


@cli.command('evaluate')
@click.option('--test_set', '-t', type=click.Choice(cli_utils.get_valid_test_sets()), required=True,
              help="test set to use.")
@click.option('--tokenizer', '-tok', type=click.Choice(['13a', 'intl', 'moses', 'plain']), default='13a',
              help="Tokenization method to use.")
@click.option('--metrics', '-m', type=str, default=cli_utils.get_valid_metrics(as_str=True),
              help=f"Comma-separated list of metrics to compute. Default: {cli_utils.get_valid_metrics(as_str=True)}")
@click.option('--analysis', '-a', is_flag=True,
              help=f"Perform word-level transformation analysis.")
def evaluate_system_output(test_set, tokenizer, metrics, analysis):
    """
    Evaluate a system output in a standard test set with appropriate automatic metrics.
    """
    # read the system output
    with click.get_text_stream('stdin', encoding='utf-8') as system_output_file:
        sys_output = system_output_file.read().splitlines()

    # get the metrics that need to be computed
    metrics = metrics.split(',')

    # get the references from the test set
    if test_set == 'turk':
        lowercase = False
        refs_sents = []
        for n in range(8):
            ref_lines = cli_utils.read_file(f"data/test_sets/turk/test.8turkers.tok.turk.{n}")
            refs_sents.append(ref_lines)

        # ref_lines = cli_utils.read_file(f"data/test_sets/turk/test.8turkers.tok.simp")
        # refs_sents.append(ref_lines)

        if 'sari' in metrics:
            # read the original sentences in plain text
            orig_sents = cli_utils.read_file("data/test_sets/turk/test.8turkers.tok.norm")

        # if 'samsa' in metrics:
        #     # read the original sentences ucca-parsed by TUPA

    if test_set == 'hsplit':
        sys_output = sys_output[:70]
        lowercase = True

        if 'sari' in metrics or 'bleu' in metrics:
            refs_sents = []
            for n in range(4):
                ref_lines = cli_utils.read_file(f"data/test_sets/hsplit/hsplit.tok.{n+1}")
                refs_sents.append(ref_lines)

        if 'sari' in metrics or 'samsa' in metrics:
            # read the original sentences in plain text
            orig_sents = cli_utils.read_file("data/test_sets/turk/test.8turkers.tok.norm")[:70]

    # compute each metric
    if 'bleu' in metrics:
        bleu_score = sacrebleu.corpus_bleu(sys_output, refs_sents, force=True, tokenize=tokenizer, lowercase=lowercase)
        click.echo(f"BLEU: {bleu_score.score}")

    if 'sari' in metrics:
        sari_score = sari.sari_corpus(orig_sents, sys_output, refs_sents, tokenizer=tokenizer, lowercase=lowercase)
        click.echo(f"SARI: {sari_score}")

    if 'samsa' in metrics:
        samsa_score = samsa.samsa_corpus(orig_sents, sys_output, tokenizer=tokenizer, verbose=True, lowercase=lowercase)
        click.echo(f"SAMSA: {samsa_score}")

    if analysis:
        word_level_analysis = annotation.analyse_wordlevel_operations(orig_sents, sys_output, refs_sents,
                                                                      verbose=False, as_str=True)
        click.echo(f"Word-level Analysis: {word_level_analysis}")


@cli.command('register')
@click.option('-n', "--name", required=True,
              help="Name of the test set. If not given, the folder name with be used.")
@click.option('-o', "--orig_file", required=True,
              help="Path of the text file with the original sentences.")
@click.option('-s', "--simp_file", required=True,
              help="Path to the text file with the simplified sentences.")
def register_test_set(name, orig_file, simp_file):
    """
    Register a new test set to pre-process it and store it locally so future computations are faster.
    """
    return


@cli.command('ranking')
@click.argument('test_set')
@click.option('-sb', "--sort_by", type=click.Choice(['bleu', 'sari', 'samsa']),
              help="Metric to use for sorting the systems' scores.")
def print_ranking(test_set, sort_by):
    """
    Print the ranking of all published system outputs in a standard test set.
    """
    return
