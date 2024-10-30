# coding: utf-8

import os

from pix2text import TextFormulaOCR, merge_line_texts


def test_mfd():
    config = dict()
    model = TextFormulaOCR.from_config(config)

    res = model.recognize(
        './docs/examples/zh1.jpg', save_analysis_res='./analysis_res.jpg',
    )
    print(res)


def test_example():
    # img_fp = './docs/examples/formula.jpg'
    img_fp = './docs/examples/mixed.jpg'
    formula_config = {
        'model_name': 'mfr-pro',
        'model_backend': 'onnx',
    }
    p2t = TextFormulaOCR.from_config(total_configs={'formula': formula_config})
    print(p2t.recognize(img_fp))
    # print(p2t.recognize_formula(img_fp))
    # outs = p2t(img_fp, resized_shape=608, save_analysis_res='./analysis_res.jpg')  # can also use `p2t.recognize(img_fp)`
    # print(outs)
    # # To get just the text contents, use:
    # only_text = merge_line_texts(outs, auto_line_break=True)
    # print(only_text)


def test_blog_example():
    img_fp = './docs/examples/mixed.jpg'

    total_config = dict(
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
    p2t = TextFormulaOCR.from_config(total_configs=total_config)
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  # The same result can also be obtained using 'p2t(img_fp)'
    print(outs)
    # If you only need the recognized text and the Latex representation, you can use the code in the following line to merge all the results
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_blog_pro_example():
    img_fp = './docs/examples/mixed.jpg'

    total_config = dict(
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
    p2t = TextFormulaOCR.from_config(total_configs=total_config)
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  # The same result can also be obtained using 'p2t(img_fp)'
    print(outs)
    # If you only need the recognized text and the Latex representation, you can use the code in the following line to merge all the results
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_mixed():
    img_fp = './docs/examples/en1.jpg'
    p2t = TextFormulaOCR.from_config()
    outs = p2t.recognize(
        img_fp, resized_shape=608, return_text=False
    )  #The same result can also be obtained using 'p2t(img_fp)'
    print(outs)
    # If you only need the recognized text and Latex representations, you can use the following lines of code to merge all the results
    only_text = merge_line_texts(outs, auto_line_break=True)
    print(only_text)


def test_example_formula():
    img_fp = './docs/examples/math-formula-42.png'
    p2t = TextFormulaOCR.from_config()
    outs = p2t.recognize_formula(img_fp)
    print(outs)


def test_example_text():
    img_fp = './docs/examples/general.jpg'
    p2t = TextFormulaOCR()
    outs = p2t.recognize_text(img_fp)
    print(outs)
