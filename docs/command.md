# Scripting tools
The Python package **pix2text** comes with a command-line tool `p2t`. [Installation](install.md) and it can be used. The following subcommands are included in `p2t`.

## predict

Use the command **`p2t predict`** to predict a single (image or PDF) file or folder (multiple images are not supported at the same time). Here are the usage instructions:

```bash
$ p2t predict -h
Usage: p2t predict [OPTIONS]

  Use Pix2Text (P2T) to predict textual information in an image or PDF file

Options:
  -l, --languages TEXT            Text-OCR recognized language codes, separated by commas, defaults to en,ch_sim
  --layout-config TEXT            Configuration information for the layout parser model, provided in JSON string format. Default values: `None` to use the default configuration
  --mfd-config TEXT               MFD model configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --formula-ocr-config TEXT       Latex-OCR mathematical formula recognition model configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --text-ocr-config TEXT          Text-OCR recognition configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --enable-formula / --disable-formula
                                  Whether to enable formula recognition, default value: Enable formulas
  --enable-table / --disable-table
                                  Whether to enable table recognition, default: Enable tables
  -d, --device TEXT               Choose to use `cpu`, `gpu`, or a specified GPU, such as `cuda:0`. Default value: cpu
  --file-type [pdf|page|text_formula|formula|text]
                                  File type to process, 'pdf', 'page', 'text_formula', 'formula', or 'text'. Default value: text_formula
  --resized-shape INTEGER         Resize the image width to this size before processing. Default value: 768
  -i, --img-file-or-dir TEXT      File path of the input image/pdf or specify a directory. [Required]
  --save-debug-res TEXT           If `save_debug_res` is set, save the debug results to the directory; default value is `None`, meaning no debug results are saved
  --rec-kwargs TEXT               kwargs for calling `.recognize()`, provided in JSON string format
  --return-text / --no-return-text
                                  Whether to return only text results, default value: return text
  --auto-line-break / --no-auto-line-break
                                  Whether to automatically determine if adjacent line results should be merged into a single line result, default value: auto line break
  -o, --output-dir TEXT           Output directory for recognized text results. Only valid when `file-type` is `pdf` or `page`. Default value: output-md
  --log-level TEXT                Log level, such as `INFO`, `DEBUG`. Default value: INFO
  -h, --help                      Show this message and exit.
```

### Example 1
Predict using the basic model:

```bash
p2t predict -l en,ch_sim --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```

It will store the recognition results (in Markdown format) in the `output-md` directory and the intermediate parsing results in the `output-debug` directory for analyzing which model mainly affects the recognition results. If you do not need to save the intermediate parsing results, you can remove the `--save-debug-res output-debug` parameter.

### Example 2

Prediction also supports using custom parameters or models. For example, using a custom model for prediction:

```bash
p2t predict -l en,ch_sim --mfd-config '{"model_name": "mfd-pro", "model_backend": "onnx"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' --rec-kwargs '{"page_numbers": [0, 1]}' --resized-shape 768 --file-type pdf -i docs/examples/test-doc.pdf -o output-md --save-debug-res output-debug
```

## start service

Use the command **`p2t serve`** to start an HTTP service to receive images (currently does not support PDFs) and return recognition results. This HTTP service is implemented based on FastAPI. Here are the usage instructions:

```bash
$ p2t serve -h
Usage: p2t serve [OPTIONS]

  Start the HTTP service.

Options:
  -l, --languages TEXT            Text-OCR recognized language codes, separated by commas, defaults to en,ch_sim
  --layout-config TEXT            Configuration information for the layout parser model, provided in JSON string format. Default values: `None` to use the default configuration
  --mfd-config TEXT               MFD model configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --formula-ocr-config TEXT       Latex-OCR mathematical formula recognition model configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --text-ocr-config TEXT          Text-OCR recognition configuration information, provided in JSON string format. Default values: `None`, to use the default configuration
  --enable-formula / --disable-formula
                                  Whether to enable formula recognition, default value: Enable formulas
  --enable-table / --disable-table
                                  Whether to enable table recognition, default: Enable tables
  -d, --device TEXT               Choose to use `cpu`, `gpu`, or a specified GPU, such as `cuda:0`. Default value: cpu
  -o, --output-md-root-dir TEXT   Root directory for Markdown output, used to store recognized text results. Only valid when `file-type` is `pdf` or `page`. Default value: output-md-root
  -H, --host TEXT                 Server host  [Default value: 0.0.0.0]
  -p, --port INTEGER              Server port  [Default value: 8503]
  --reload                        Whether to reload the server when the code changes
  --log-level TEXT                Log level, such as `INFO`, `DEBUG`. Default value: INFO
  -h, --help                      Show this message and exit.
```

### Example 1
Predict using the basic model:

```bash
p2t serve -l en,ch_sim -H 0.0.0.0 -p 8503
```

### Example 2

The service also supports using custom parameters or models when started. For example, using a custom model for prediction:

```bash
p2t serve -l en,ch_sim --mfd-config '{"model_name": "mfd-pro", "model_backend": "onnx"}' --formula-ocr-config '{"model_name":"mfr-pro","model_backend":"onnx"}' --text-ocr-config '{"rec_model_name": "doc-densenet_lite_666-gru_large"}' -H 0.0.0.0 -p 8503
```

### Service Invocation

#### Python
After starting, you can use the following method to invoke the command (Python):

```py3
import requests

url = 'http://0.0.0.0:8503/pix2text'

image_fp = 'docs/examples/page2.png'
data = {
    "file_type": "page",
    "resized_shape": 768,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}
files = {
    "image": (image_fp, open(image_fp, 'rb'), 'image/jpeg')
}

r = requests.post(url, data=data, files=files)

outs = r.json()['results']
out_md_dir = r.json()['output_dir']
if isinstance(outs, str):
    only_text = outs
else:
    only_text = '\n'.join([out['text'] for out in outs])
print(f'{only_text=}')
print(f'{out_md_dir=}')
```

#### Curl

You can also use curl to invoke the service:

```bash
curl -X POST \
  -F "file_type=page" \
  -F "resized_shape=768" \
  -F "embed_sep= $,$ " \
  -F "isolated_sep=$$\n, \n$$" \
  -F "image=@docs/examples/page2.png;type=image/jpeg" \
  http://0.0.0.0:8503/pix2text
```
