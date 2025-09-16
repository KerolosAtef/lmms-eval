import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
from loguru import logger as eval_logger
import ast
import datetime
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Union,Tuple, Any, Optional
import torch
import decord
import cv2
import datasets
import numpy as np
import requests
import yaml
import pysrt
from loguru import logger as eval_logger
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from huggingface_hub import hf_hub_download
from openai import OpenAI

# InfiniBench skills categorization
OPEN_ENDED_SKILLS = ['summarization', 'spoiler_questions', 'deep_context_understanding', 'linking_multiple_events']
MCQ_SKILLS = ['character_actions', 'scene_transitions', 'choronological_understanding', 'global_appearance']
ALL_SKILLS = OPEN_ENDED_SKILLS + MCQ_SKILLS
n_qa_per_skill = {}
for split in ['train', 'validation', 'test']:
    n_qa_per_skill[split] = {}
    for skill in ALL_SKILLS:
        n_qa_per_skill[split][skill] = 0

# OpenAI API configuration for GPT evaluation
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-mini")
API_TYPE = os.getenv("API_TYPE", "openai")

FRAME_FACTOR = 2
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def download_and_extract_infinibench_videos(config_file , split: str = "validation"):
    """
    Download and extract InfiniBench videos for the specified split.
    
    Args:
        cache_dir: Directory to store downloaded and extracted videos
        split: Dataset split ('train', 'validation', 'test')
    """
    cache_dir = config_file.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
    cache_path = Path(cache_dir)
    videos_dir = cache_path / f"{split}"
    
    # Check if videos already extracted
    expected_dirs = ["TV_shows", "Movies"] 
    if all((videos_dir / d).exists() for d in expected_dirs):
        eval_logger.info(f"Videos for {split} split already extracted in {videos_dir}")
        return str(videos_dir)
    
    eval_logger.info(f"Downloading InfiniBench {split} videos...")
    
    # Get list of video part files for this split
    repo_files = list_repo_files(config_file['dataset_path'], repo_type="dataset")
    # repo_files = list_repo_files("Vision-CAIR/InfiniBench", repo_type="dataset")
    # Find video parts for the specified split
    video_parts = [f for f in repo_files if f.startswith(f"{split}/{split}_videos.tar.gz.part_")]
    video_parts.sort()  # Ensure correct order
    
    if not video_parts:
        eval_logger.warning(f"No video parts found for split: {split}")
        return str(videos_dir)
    
    eval_logger.info(f"Found {len(video_parts)} video parts for {split} split")
    
    # Download all parts
    downloaded_parts = []
    for part_file in video_parts:
        eval_logger.info(f"Downloading {part_file}...")
        local_path = hf_hub_download(
            # repo_id="Vision-CAIR/InfiniBench",
            repo_id=config_file['dataset_path'],
            repo_type="dataset", 
            filename=part_file,
            cache_dir=cache_dir,
            local_dir=cache_path,
            local_dir_use_symlinks=False
        )
        downloaded_parts.append(local_path)
    
    # Reassemble the tar.gz file
    tar_file = cache_path / f"{split}_videos.tar.gz"
    eval_logger.info(f"Reassembling {tar_file}...")
    
    with open(tar_file, "wb") as output:
        for part_path in downloaded_parts:
            with open(part_path, "rb") as part:
                output.write(part.read())
    
    # Extract the tar.gz file 
    eval_logger.info(f"Extracting {tar_file} to {videos_dir}...")
    
    eval_logger.info("This may take a while...")
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=videos_dir)
    
    # Clean up
    tar_file.unlink()  # Remove the reassembled tar file
    for part_path in downloaded_parts:
        Path(part_path).unlink()  # Remove individual parts
    
    eval_logger.info(f"Videos extracted successfully to {videos_dir}")
    return str(videos_dir)


def load_config_file(yaml_filename: str) -> str:
    with open(Path(__file__).parent / yaml_filename, "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
        config = yaml.safe_load("".join(safe_data))
    return config
config_file_val = load_config_file("infinibench_validation.yaml")
config_file_test = load_config_file("infinibench_test.yaml")
config_file_train = load_config_file("infinibench_train.yaml")

Download_Videos=False
used_split=""
for split in ['train', 'validation', 'test']:
    data_obj=datasets.load_dataset(config_file_val.get("dataset_path"), split=split, cache_dir=config_file_val.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache"))
    # print(f"{split} split has {len(data_obj)} samples")
    for item in data_obj:
        skill_name = item.get("skill_name", "unknown_skill")
        if skill_name in ALL_SKILLS:
            n_qa_per_skill[split][skill_name] += 1
    
    # map the split name to be consistent
    if split == 'validation':
        split='dev'
        n_qa_per_skill['dev'] = n_qa_per_skill['validation']
        n_qa_per_skill.pop('validation')
    # print(f"{split} split QA counts for summarization : {n_qa_per_skill[split]['summarization']}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def time_to_milliseconds(time_str):
    # Convert time format "hh:mm:ss.sss" to milliseconds
    h, m, s = map(float, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * 1000)
    
def read_srt_subtitles(subtitle_path):
    # choose only the subtitles that are within the video frames time
    subs = pysrt.open(subtitle_path)
    subtitles = []
    for sub in subs:
        start_time = time_to_milliseconds(str(sub.start.to_time()))
        end_time = time_to_milliseconds(str(sub.end.to_time()))
        subtitles.append((start_time, end_time, sub.text))
    return subtitles

        
def match_subtitles_to_frames(subtitles, frames_time):
    # Match each subtitle to the corresponding frame
    matched_subtitles = ""
    for frame_time in frames_time:
        subtitle = find_exact_frame_subtitles(subtitles, frame_time*1000)
        if subtitle:
            matched_subtitles += subtitle + "\n"
    return matched_subtitles     

    
def find_exact_frame_subtitles(subtitles, frame_time):
    left, right = 0, len(subtitles) - 1
    all_matched_subtitles = []
    while left <= right:
        mid = (left + right) // 2
        start, end, subtitle_text = subtitles[mid]
        # print("Mid start end sub ",mid,start,end,subtitle_text)
        if start <= frame_time <= end:
            all_matched_subtitles.append(subtitle_text)
            left = mid + 1
        elif frame_time < start:
            right = mid - 1
        else:
            left = mid + 1
    if len(all_matched_subtitles) > 0:
        return " ".join(all_matched_subtitles)
    return None  # If no subtitle is found


def prepare_prompt(question, video_path,subtitle_path,skill_name,n_frames=128):
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = round_by_factor(n_frames, FRAME_FACTOR)    
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    frames_timestamp = [frame_idx / video_fps for frame_idx in idx]
    if subtitle_path is not None and os.path.exists(subtitle_path):
        subtitles=read_srt_subtitles(subtitle_path)
        matched_subtitles = match_subtitles_to_frames(subtitles, frames_timestamp)
        model_input = f"Given this video's subtitles : {matched_subtitles}" +question
    else:
        model_input = question
    return model_input 

def infinibench_doc_to_visual(doc):
    global Download_Videos
    global used_split
    if not Download_Videos:
        if 'split' in doc and doc['split'] == 'train':
            download_and_extract_infinibench_videos(config_file_train, split="train")
            Download_Videos=True
        elif 'split' in doc and doc['split'] == 'dev':
            download_and_extract_infinibench_videos(config_file_val, split="validation")
            Download_Videos=True
        else:
            download_and_extract_infinibench_videos(config_file_test, split="test")
            Download_Videos=True
    
    if 'split' in doc and doc['split'] == 'train':
        cache_dir_train = config_file_train.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        video_path = os.path.join(cache_dir_train,"train", doc["video_path_mp4"])
        used_split='train'
    elif 'split' in doc and doc['split'] == 'dev':
        cache_dir_val = config_file_val.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        video_path = os.path.join(cache_dir_val, "validation",doc["video_path_mp4"])
        used_split='dev'
    else:
        cache_dir_test = config_file_test.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        video_path = os.path.join(cache_dir_test,"test", doc["video_path_mp4"])
        used_split='test'
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def infinibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    question = doc["question"]
    skill_name = doc.get("skill_name", "unknown_skill")
    if skill_name in MCQ_SKILLS:
        answer_idx = doc['answer_idx']
        question = doc['question'] + "\nOptions are:\n"
        options = doc.get('options', [])
        options = ast.literal_eval(options) if isinstance(options, str) else options
        for idx, option in enumerate(options):
            option_letter = chr(65 + idx)  # Convert index to letter (0 -> A, 1 -> B, etc.)
            question += f"({option_letter}) {option}\n"
    if skill_name in MCQ_SKILLS:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt_mcq", "")
    else:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt_open_ended", "")
    full_prompt = pre_prompt + question + post_prompt
    n_frames=lmms_eval_specific_kwargs.get("n_frames", 128)
    use_subtitle=lmms_eval_specific_kwargs.get("use_subtitle", True)
    split= doc.get("split")
    if split == 'train':
        cache_dir_train = config_file_train.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        subtitle_path = os.path.join(cache_dir_train,"train", doc ['video_subtitles'])
    elif split == 'dev':
        cache_dir_val = config_file_val.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        subtitle_path = os.path.join(cache_dir_val, "validation",doc ['video_subtitles'])
    else:
        cache_dir_test = config_file_test.get("dataset_kwargs", {}).get("cache_dir", "./infinibench_cache")
        subtitle_path = os.path.join(cache_dir_test,"test", doc ['video_subtitles'])
    if not use_subtitle:
        subtitle_path = None
    video_path = infinibench_doc_to_visual(doc)[0]
    full_prompt = prepare_prompt(full_prompt, video_path, subtitle_path, skill_name,n_frames)
    return full_prompt


def infinibench_doc_to_target(doc):
    """
    Extract target answer from document.
    
    Args:
        doc: Document containing answer information
        
    Returns:
        str or int: Target answer (string for open-ended, int for MCQ)
    """
    
    # For MCQ questions
    if "answer_idx" in doc:
        return doc["answer_idx"]
    
    # For open-ended questions
    if "answer" in doc:
        return doc["answer"]
    
    # Fallback
    return ""

def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    else:
        return s
def infinibench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case infinibench score), value: metric value
    """
    pred = results[0]
    skill_name = doc.get("skill_name", "unknown_skill")
    split= doc.get("split")
    
    if skill_name in MCQ_SKILLS:
        answer = chr(65 + int(doc['answer_idx']))
        pred_ans = re.findall(r'\((.*?)\)', pred)
        if len(pred_ans) > 0:
            pred_ans = pred_ans[0]
        else:
            eval_logger.warning(f"Warning: No option found in model prediction: {pred}. Using full prediction as answer for question id: {doc.get('question_id', 'unknown')}")
            pred_ans = pred
    else:
        answer = doc['answer']
        pred_ans = pred.replace('Answer :', '').strip()
    split= doc.get("split")
    if split == 'train':
        lmms_eval_specific_kwargs = config_file_train.get("lmms_eval_specific_kwargs", {})
    elif split == 'dev':
        lmms_eval_specific_kwargs = config_file_val.get("lmms_eval_specific_kwargs", {})
    else:
        lmms_eval_specific_kwargs = config_file_test.get("lmms_eval_specific_kwargs", {})
    input_prompt = infinibench_doc_to_text(doc, lmms_eval_specific_kwargs)
    data_dict = {
        "question_id": doc.get("question_id", "unknown"), 
        "task_type": skill_name, 
        "pred": pred_ans, 
        "answer": answer, 
        "question": input_prompt,
        "raw_model_output": pred,
        "split": split
    }

    # Return data structure that the aggregation function expects
    return {"Overall_score": data_dict}

def gpt_score_wrapper(qa: Dict[str, Any], max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Evaluate a question-answer pair using GPT-4o-mini with comprehensive error handling and retry logic.
    
    Args:
        qa (Dict[str, Any]): Question-answer dictionary containing 'question', 'answer', and 'pred' keys
        max_retries (int): Maximum number of retry attempts for failed requests
        retry_delay (float): Base delay between retries in seconds (with exponential backoff)
        
    Returns:
        Dict[str, Any]: Updated QA dictionary with 'gpt_score' and 'gpt_justification' fields
    """
    if qa.get("gpt_score") is not None:
        eval_logger.debug(f"Skipping already scored QA pair")
        return qa
    
    # Check if client is available
    if client is None:
        eval_logger.error("OpenAI client not initialized. Please check your API key.")
        qa["gpt_score"] = None
        qa["gpt_justification"] = "OpenAI client not available"
        return qa
    
    # Validate required fields
    required_fields = ["question", "answer", "pred"]
    missing_fields = [field for field in required_fields if field not in qa]
    if missing_fields:
        eval_logger.warning(f"Missing required fields: {missing_fields}")
        qa["gpt_score"] = None
        qa["gpt_justification"] = f"Missing fields: {missing_fields}"
        return qa
    
    question = qa["question"]
    answer = qa["answer"]
    pred = qa["pred"]
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent and fair evaluator AI that specializes in assessing the correctness and semantic alignment "
                            "between ground truth answers and predicted responses for question-answering tasks, including those based on video content.\n\n"
                            "Your role is to evaluate how well a predicted answer matches the correct (reference) answer based on the following detailed criteria:\n"
                            "------\n"
                            "## EVALUATION INSTRUCTIONS:\n"
                            "- Focus on **semantic similarity**, **factual correctness**, and **completeness**.\n"
                            "- Accept paraphrases, synonyms, or rephrasings **as valid**, as long as they preserve the original meaning.\n"
                            "- **Do not penalize** for stylistic differences or changes in tone, unless they impact factual accuracy.\n"
                            "- **Penalize** if:\n"
                            "  - The predicted answer omits **key factual elements** present in the correct answer.\n"
                            "  - The prediction includes **hallucinated content** or unfounded details.\n"
                            "  - The prediction **contradicts** the correct answer.\n"
                            "- Use human-like judgment: apply reasoning beyond surface text similarity.\n"
                            "- When uncertain, provide a **conservative but fair** score.\n"
                            "- Use a scoring scale from **0 (completely incorrect)** to **10 (perfect match)**.\n"
                            "## OUTPUT FORMAT:\n"
                            "Return a JSON object with **two fields**:\n"
                            '- "score": an integer from 0 to 10\n'
                            '- "justification": a concise explanation (1-3 sentences) of your reasoning\n\n'
                            "### Example Output:\n"
                            "{\n"
                            '  "score": 7,\n'
                            '  "justification": "The predicted answer captures the main idea, but it omits some key details about the setting described in the correct answer."\n'
                            "}\n"
                            "------\n"
                            "Be fair, consistent, and concise. Follow the format exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Please return your evaluation in the specified JSON format with both a score and a justification."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=500,
                timeout=30,  # 30 second timeout
            )
            
            response = completion.choices[0].message.content
            if isinstance(response, str):
                response_json = ast.literal_eval(response)
            
            # Validate response format
            if "score" not in response_json or "justification" not in response_json:
                raise ValueError(f"Invalid response format: {response_json}")
            
            # Validate score range
            score = response_json["score"]
            if not isinstance(score, int) or score < 0 or score > 10:
                raise ValueError(f"Invalid score: {score}. Must be integer between 0-10")
            
            qa["gpt_score"] = score
            qa["gpt_justification"] = response_json["justification"]
            eval_logger.debug(f"Successfully scored QA pair with score: {score}")
            
            # Success - break out of retry loop
            break
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if this is a retryable error
            retryable_errors = [
                "connection error", "timeout", "rate limit", "server error", 
                "503", "502", "500", "429", "network", "connection"
            ]
            
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if attempt < max_retries and is_retryable:
                # Exponential backoff with jitter
                delay = retry_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
                eval_logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                eval_logger.info(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            else:
                # Final attempt failed or non-retryable error
                if attempt == max_retries:
                    eval_logger.error(f"Failed to score QA pair after {max_retries + 1} attempts. Last error: {e}")
                else:
                    eval_logger.error(f"Non-retryable error scoring QA pair: {e}")
                
                qa["gpt_score"] = None
                qa["gpt_justification"] = f"Evaluation failed: {str(e)}"
                break
    return qa
def mcq_accuracy(res):
    global n_qa_per_skill
    if len(res) == 0:
        return 0
    total_correct = 0
    skill_name=res[0].get("task_type", "unknown_skill")
    split=res[0].get("split", "unknown_split")
    total_qa = n_qa_per_skill.get(split, {}).get(skill_name, 0)
    for r in res:
        # total_answered += 1
        # For MCQ, compare the predicted letter with the ground truth letter
        total_correct += str(r["pred"]) == str(r["answer"])
    accuracy = 100 * (total_correct / total_qa) if total_qa > 0 else 0
    return accuracy

def infinibench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    if len(results) > 0:
        split = results[0].get("split", "unknown_split")
    else:
        split = "unknown_split"
    global n_qa_per_skill
    tasks = {}
    for skill in ALL_SKILLS:
        tasks[skill] = []
    
    for result in results:
        task_type = result["task_type"]
        if task_type in OPEN_ENDED_SKILLS:
            result = gpt_score_wrapper(result)
            tasks[task_type].append(result)
        elif task_type in MCQ_SKILLS:
            tasks[task_type].append(result)
        else:
            eval_logger.warning(f"Unknown task type: {task_type}, skipping...")
    
    # Save all the results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "infinibench_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"infinibench_detailed_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=4)
    eval_logger.info(f"Detailed results saved to {output_path}")
    
    task_category_scores = {}
    total_open_ended_score = 0
    total_mcq_accuracy = 0
    
    for task_cate, res in tasks.items():
        # if len(res) == 0:
        #     eval_logger.warning(f"No results found for task category: {task_cate}")
        #     continue
            
        if task_cate in MCQ_SKILLS:
            accuracy = mcq_accuracy(res)
            task_category_scores[task_cate] = accuracy
            total_mcq_accuracy += accuracy
        else:
            # For open-ended questions
            total_score = sum(r["gpt_score"] for r in res if r["gpt_score"] is not None)
            # total_answered = sum(1 for r in res if r["gpt_score"] is not None)
            # category_score = total_score / total_answered if total_answered > 0 else 0
            # split=res.get("split", "unknown_split")
            total_qa= n_qa_per_skill.get(split, {}).get(task_cate, 0)
            category_score = total_score / total_qa if total_qa > 0 else 0
            task_category_scores[task_cate] = category_score
            total_open_ended_score += category_score
    
    average_open_ended_score = total_open_ended_score / len(OPEN_ENDED_SKILLS) if OPEN_ENDED_SKILLS else 0
    average_mcq_accuracy = total_mcq_accuracy / len(MCQ_SKILLS) if MCQ_SKILLS else 0
    
    # Print professional results table
    print("\n" + "="*95)
    print("                            INFINIBENCH EVALUATION RESULTS")
    print("="*95)
    
    # Header
    print(f"{'Skill Category':<35} {'Type':<15} {'Total QA':<12} {'Score':<15}")
    print("-" * 95)
    
    # MCQ Skills
    for skill in MCQ_SKILLS:
        if skill in task_category_scores:
            total_qa = n_qa_per_skill.get(split, {}).get(skill, 0)
            score_str = f"{task_category_scores[skill]:.2f}%"
            print(f"{skill.replace('_', ' ').title():<35} {'MCQ':<15} {total_qa:<12} {score_str:<15}")
    
    # Open-ended Skills  
    for skill in OPEN_ENDED_SKILLS:
        if skill in task_category_scores:
            total_qa = n_qa_per_skill.get(split, {}).get(skill, 0)
            score_str = f"{task_category_scores[skill]:.2f}/10"
            print(f"{skill.replace('_', ' ').title():<35} {'Open-ended':<15} {total_qa:<12} {score_str:<15}")
    
    print("-" * 95)
    
    # Calculate totals
    total_mcq_qa = sum(n_qa_per_skill.get(split, {}).get(skill, 0) for skill in MCQ_SKILLS)
    total_open_ended_qa = sum(n_qa_per_skill.get(split, {}).get(skill, 0) for skill in OPEN_ENDED_SKILLS)
    total_all_qa = total_mcq_qa + total_open_ended_qa
    
    # Summary statistics
    print(f"{'Average MCQ Accuracy':<35} {'Summary':<15} {total_mcq_qa:<12} {average_mcq_accuracy:.1f}%")
    print(f"{'Average Open-ended Score':<35} {'Summary':<15} {total_open_ended_qa:<12} {average_open_ended_score:.1f}/10")
    
    print("=" * 95)
    overall_score_val = 0.5 * (average_open_ended_score / 10) + 0.5 * (average_mcq_accuracy / 100)
    print(f"{'OVERALL SCORE':<35} {'Final':<15} {total_all_qa:<12} {overall_score_val * 100:.2f}/100")
    print("=" * 95)
    print()
    
    overall_score = overall_score_val
    return overall_score * 100  # Scale to 0-100
