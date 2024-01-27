from clarifai.client.model import Model
import os

inference_params = dict(temperature=0.2, max_tokens=250)


def getGeminiResponse(input):
    gemini_inference_params = dict(
        temperature=0.2, top_k=50, top_p=0.95, max_tokens=2048
    )
    model_prediction = Model(
        "https://clarifai.com/gcp/generate/models/gemini-pro"
    ).predict_by_bytes(
        input.encode(), input_type="text", inference_params=gemini_inference_params
    )

    response = model_prediction.outputs[0].data.text.raw
    return response


def translate_webpage(html_file, lang):
    if not os.path.exists('output'):
        os.mkdir('output')
    output_path = os.path.join('output', html_file)
    html_code = None
    if os.path.exists(html_file):
        with open(html_file, 'r') as f:
            html_code = f.read()
    else:
        raise FileNotFoundError(f"The file '{html_file}' doesn't exist")
    translated_code = getGeminiResponse(f"""
translate the raw text in the following code to {lang},
make sure to keep the tags and code as is, reply with only the full code.\n
code:\n
{html_code}
""")
    extracted_code = None
    if '```' in translated_code[0:3]:
        start_index = translated_code.find("```html") + 7
        end_index = translated_code.rfind("```")

        extracted_code = translated_code[start_index:end_index]
        if not extracted_code:
            raise RuntimeError("LLM didn't return code")
    with open(output_path, "w", encoding="utf-8") as html_output_file:
        html_output_file.write(
            extracted_code if extracted_code else translated_code)

    return f"Absolute path for the generated HTML file: {os.path.realpath(output_path)}"


print(translate_webpage(input("Enter the path to the html file below:\n"),
      input("Enter the language you want the webpage translated to below:\n")))
