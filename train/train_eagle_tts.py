# -*- coding: utf-8 -*-
import aiohttp
import asyncio
import argparse
import json
import time
from tqdm import tqdm
import os
from datetime import datetime
from collections import defaultdict

headers = {
    "Content-Type": "application/json"
}

async def process_request(session, data, progress_bar):
    try:
        json_d = json.loads(data)
        msg = [{
            "role": "user",
            "content": json_d['input'].replace('REDACTED_SPECIAL_TOKEN', '\n\n').replace('REDACTED_SPECIAL_TOKEN', '')
        }]
        
        json_data = {
            "model": "deepseek-r1",
            "stream": False,
            "messages": msg
        }

        async with session.post(
            'http://deepseek.sankuai.com/v1/chat/completions',
            headers=headers,
            json=json_data,
            timeout=500
        ) as response:
            response_text = await response.text()
            res_j = json.loads(response_text)
            res = res_j['choices'][0]['message']['content'].replace('```json','').replace('```','').replace('\n','')
            d = json.loads(res)
            out_json = json.dumps({
                "id": json_d['target'],
                "name": d['name'].strip(),
                'recommendation': d['recommendation'].strip()
            }, ensure_ascii=False)
            progress_bar.update(1)
            return out_json, None, None
            
    except Exception as e:
        progress_bar.update(1)
        error_msg = f"Error: {str(e)}"
        if 'response_text' in locals():
            error_msg += f"\nResponse: {response_text}"
        return None, data, error_msg

async def process_batch(session, batch, progress_bar):
    tasks = [process_request(session, data, progress_bar) for data in batch]
    return await asyncio.gather(*tasks)

async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-fpi", "--file-path-in", type=str, default="", help="输入路径")
    argparser.add_argument("-fpo", "--file-path-out", type=str, default="", help="输出路径")
    argparser.add_argument("-qps", "--queries-per-second", type=int, default=10, help="每秒请求数")
    
    args = argparser.parse_args()
    args_dict = vars(args)
    print("所有命令行参数：")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    
    with open(args.file_path_in, 'r') as f:
        in_data = f.readlines()
    
    # 创建进度条
    progress_bar = tqdm(total=len(in_data), desc="请求进度")
    
    # 创建异步会话
    timeout = aiohttp.ClientTimeout(total=500)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        all_results = []
        # 按批次处理请求
        for i in range(0, len(in_data), args.queries_per_second):
            batch = in_data[i:i + args.queries_per_second]
            results = await process_batch(session, batch, progress_bar)
            all_results.extend(results)
            # 等待1秒
            await asyncio.sleep(1)
    
    # 统计结果
    success_count = 0
    failure_count = 0
    error_types = defaultdict(int)
    
    # 处理结果
    with open(args.file_path_out, 'w') as g:
        with open(args.file_path_out+'-fail', 'w') as m:
            for result, failed_data, error_msg in all_results:
                if result:
                    g.write(result + '\n')
                    success_count += 1
                else:
                    m.write(failed_data)
                    failure_count += 1
                    if error_msg:
                        error_type = error_msg.split('\n')[0]  # 只取第一行作为错误类型
                        error_types[error_type] += 1
    
    progress_bar.close()
    
    # 打印统计信息
    print("\n请求统计信息:")
    print(f"总请求数: {len(in_data)}")
    print(f"成功请求数: {success_count}")
    print(f"失败请求数: {failure_count}")
    print(f"成功率: {(success_count/len(in_data))*100:.2f}%")
    
    if failure_count > 0:
        print("\n失败原因统计:")
        for error_type, count in error_types.items():
            print(f"{error_type}: {count}次")

if __name__ == "__main__":
    asyncio.run(main()) 