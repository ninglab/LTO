import re
from prompt_pool import ANSWER_PREFIX


class AnswerParsing:
    def __init__(self, dataset):
        self.dataset = dataset

    def dataset_parse(self, pred, true, sample):
        if self.dataset == "mgsm" or self.dataset == "gsm8k":
            extracted_answer, binary = self.mgsm_parse(ANSWER_PREFIX["en"], pred, true)
        elif self.dataset == "math":
            extracted_answer, binary = self.math_parse(ANSWER_PREFIX["en"], pred, true)
        elif self.dataset == "commonsenseqa":
            extracted_answer, binary = self.commonsenseqa_parse(pred, true)
        elif self.dataset == "theoremqa":
            extracted_answer, binary = self.theoremqa_parse(sample["answer_type"], pred, true)
        elif self.dataset == "mmmlu":
            extracted_answer, binary = self.mmmlu_parse(pred, true)
        elif self.dataset == "belebele":
            extracted_answer, binary = self.belebele_parse(pred, true)

        return extracted_answer, binary
        

    def extract_boxed_content(self, text):
        pattern = re.compile(r'\\boxed{')
        matches = pattern.finditer(text)
        results = []
        for match in matches:
            start_pos = match.end()
            brace_count = 1
            i = start_pos
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                results.append(text[start_pos:i-1])
        return results


    def mgsm_parse(self, answer_prefix, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        if answer_prefix not in pred:
            return None, False

        answer_text = pred.split(answer_prefix)[-1].strip()
        numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
        extracted_answer = numbers[-1].rstrip(".") if numbers else ""

        if "." in extracted_answer:
            extracted_answer = extracted_answer.rstrip("0").rstrip(".")
        true = true.replace(",", "")
        extracted_answer = extracted_answer.replace(",", "")

        return extracted_answer, true == extracted_answer


    def math_parse(self, answer_prefix, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        extracted_answer = self.extract_boxed_content(pred)
        extracted_answer = extracted_answer[0] if extracted_answer else None
        if extracted_answer:
            extracted_answer = extracted_answer.replace(" ", "")
            true = true.replace(" ", "")

        return extracted_answer, true == extracted_answer


    def mmmlu_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None
            
        return extracted_answer, true == extracted_answer


    def commonsenseqa_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-E])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None
        
        return extracted_answer, true == extracted_answer


    def belebele_parse(self, pred, true):
        alpha_map = ["", "A", "B", "C", "D"]
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None

        return extracted_answer, alpha_map[int(true)] == extracted_answer

    def theoremqa_parse(self, answer_type, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        answer_text = pred.split("answer is")[-1].strip()
        pred = re.sub(r'\<.*\>', '', answer_text)

        if answer_type == "bool":
            if "True" in pred or "true" in pred:
                extracted_answer = "True"
            elif "False" in pred or "false" in pred:
                extracted_answer = "False"
            else:
                extracted_answer = None
        elif answer_type == "integer" or answer_type == "float":
            numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
            extracted_answer = numbers[-1].rstrip(".") if numbers else ""
        elif answer_type == "list of integer" or answer_type == "list of float":
            match = re.search(r"\[(.*?)\]",  pred)
            extracted_answer = match.group(1) if match else None  
            true = true[1:-1]
        elif answer_type == "option":
            match = re.search(r"\(([a-d])\)", pred)
            extracted_answer = match.group(1) if match else None
            true = true[1:-1]
        
        return extracted_answer, true == extracted_answer