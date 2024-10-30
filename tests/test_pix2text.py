# coding: utf-8

import os

from pix2text import Pix2Text, set_logger

set_logger()


def test_recognize_pdf():
    pdf_fn = '1804.07821'
    img_fp = f'./docs/examples/{pdf_fn}.pdf'
    text_formula_config = dict(
        languages=('en', 'ch_sim'),
        mfd=dict(  # Declare the initialization parameters of the MFD
            model_path=os.path.expanduser(
                '~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
        formula=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.1/mfr-pro-onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
        text=dict(
            rec_model_name='doc-densenet_lite_666-gru_large',
            rec_model_backend='onnx',
            rec_model_fp=os.path.expanduser(
                '~/.cnocr/2.3/doc-densenet_lite_666-gru_large/cnocr-v2.3-doc-densenet_lite_666-gru_large-epoch=005-ft-model.onnx'
                # noqa
            ),  # Note: Modify it to the path where your model file is stored
        ),
    )
    total_config = {
        'layout': {'scores_thresh': 0.45},
        'text_formula': text_formula_config,
    }
    p2t = Pix2Text.from_config(total_configs=total_config, enable_formula=True)
    out_md = p2t.recognize_pdf(
        img_fp,
        page_numbers=[0, 7, 8],
        table_as_image=True,
        save_debug_res=f'./outputs-{pdf_fn}',
    )
    out_md.to_markdown('page-output')
    # print(out_page)
    # out_page.to_markdown('page-output')


def test_recognize_page():
    # img_fp = './docs/examples/formula.jpg'
    img_fp = './docs/examples/page3.png'
    # img_fp = './docs/examples/mixed.jpg'
    total_config = {
        'layout': {'scores_thresh': 0.45},
        'text_formula': {
            'formula': {
                'model_name': 'mfr',
                'model_backend': 'onnx',
                'more_model_configs': {'provider': 'CPUExecutionProvider'},
            }
        },
    }
    p2t = Pix2Text.from_config(total_configs=total_config)
    out_page = p2t.recognize_page(
        img_fp,
        page_id='test_page_1',
        title_contain_formula=False,
        text_contain_formula=False,
        save_debug_res='./outputs',
    )
    # print(out_page)
    out_page.to_markdown('page-output')


def test_spell_checker():
    from spellchecker import SpellChecker

    spell = SpellChecker()

    # 找到拼写错误
    misspelled = spell.unknown(["speci-fied"])

    for word in misspelled:
        # Get the one `most likely` answer
        print('word:', word, ' ->', spell.correction(word))

        # Get a list of `likely` options
        print('suggestions:', spell.candidates(word))


def test_blog_example():
    img_fp = './docs/examples/mixed.jpg'

    text_formula_config = dict(
        mfd=dict(  # Declare the initialization parameters of the MFD
            model_path=os.path.expanduser(
                '~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
        formula=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.1/mfr-pro-onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
    )
    total_config = {
        'layout': {'scores_thresh': 0.2},
        'text_formula': text_formula_config,
    }
    p2t = Pix2Text.from_config(total_configs=total_config)
    outs = p2t.recognize_page(
        img_fp,
        resized_shape=608,
        page_id='test_page_2',
        save_layout_res='./layout_res-mixed.jpg',
    )  # The same result can also be obtained using 'p2t(img_fp)'
    print(outs)


def test_blog_pro_example():
    img_fp = './docs/examples/mixed.jpg'

    text_formula_config = dict(
        languages=('en', 'ch_sim'),
        mfd=dict(  # Declare the initialization parameters of the MFD
            model_path=os.path.expanduser(
                '~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
        formula=dict(
            model_name='mfr-pro',
            model_backend='onnx',
            model_dir=os.path.expanduser(
                '~/.pix2text/1.1/mfr-pro-onnx'
            ),  # Note: Modify it to the path where your model file is stored
        ),
        text=dict(
            rec_model_name='doc-densenet_lite_666-gru_large',
            rec_model_backend='onnx',
            rec_model_fp=os.path.expanduser(
                '~/.cnocr/2.3/doc-densenet_lite_666-gru_large/cnocr-v2.3-doc-densenet_lite_666-gru_large-epoch=005-ft-model.onnx'
                # noqa
            ),  # Note: Modify it to the path where your model file is stored
        ),
    )
    p2t = Pix2Text.from_config(total_configs={'text_formula': text_formula_config})
    outs = p2t.recognize_page(
        img_fp, resized_shape=608, page_id='test_page_3'
    )  # The same result can also be obtained using 'p2t(img_fp)'
    print(outs)


def test_example_mixed():
    img_fp = './docs/examples/en1.jpg'
    p2t = Pix2Text.from_config()
    outs = p2t.recognize_page(
        img_fp, resized_shape=608, page_id='test_page_4'
    )  # The same result can also be obtained using 'p2t(img_fp)'
    print(outs)


def test_example_formula():
    img_fp = './docs/examples/math-formula-42.png'
    p2t = Pix2Text.from_config()
    outs = p2t.recognize_formula(img_fp)
    print(outs)


def test_example_text():
    img_fp = './docs/examples/general.jpg'
    p2t = Pix2Text(enable_formula=False)
    outs = p2t.recognize_text(img_fp)
    print(outs)
