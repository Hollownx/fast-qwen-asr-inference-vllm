# 流式ASR客户端 - 用于测试目的

import asyncio
import json
import argparse
import sys
import websockets
from datetime import datetime

import os
import time

# 每次发送的音频块大小（字节）：16kHz采样率、16位单声道下约20ms的音频数据
CHUNK_BYTES = 640  # 20ms at 16kHz 16-bit mono

async def sender(ws, pcm_path: str):
    """
    发送音频数据到服务器
    先发送配置消息，然后分块发送音频数据，最后发送停止消息
    """
    # 握手/配置：发送开始消息，告知服务器音频格式
    await ws.send(json.dumps({
        "type": "start",
        "format": "pcm_s16le",  # PCM 16位小端序格式
        "sample_rate_hz": 16000,  # 采样率16kHz
        "channels": 1  # 单声道
    }))

    print(f"Streaming {pcm_path}...")
    # 以二进制模式打开音频文件
    with open(pcm_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)  # 读取固定大小的音频块
            if not chunk:
                break  # 文件读取完毕
            await ws.send(chunk)  # 发送音频块
            await asyncio.sleep(0)  # 让出控制权，确保接收器可以处理消息

    # 发送停止消息，通知服务器音频传输完成
    await ws.send(json.dumps({"type": "stop"}))
    print("Finished sending audio.")

async def receiver(ws):
    """
    接收服务器返回的转录结果
    处理ready、partial、final、error等不同类型的消息
    """
    async for message in ws:
        try:
            # 获取当前时间戳（精确到毫秒）
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            evt = json.loads(message)  # 解析JSON消息
            msg_type = evt.get('type')  # 消息类型
            text = evt.get('text', '')  # 转录文本
            lang = evt.get('language', '')  # 检测到的语言
            
            if msg_type == 'ready':
                # 服务器就绪消息
                print(f"[{timestamp}] [Server Ready]")
            elif msg_type == 'partial':
                # 部分结果：使用\r覆盖当前行，保持输出整洁
                sys.stdout.write(f"\r[{timestamp}] [Partial] ({lang}): {text}")
                sys.stdout.flush()
            elif msg_type == 'final':
                # 最终结果：换行显示完整转录文本
                print(f"\n[{timestamp}] [Final] ({lang}): {text}")
            elif msg_type == 'error':
                # 错误消息
                print(f"\n[{timestamp}] [Error]: {evt.get('message')}")
            else:
                # 未知消息类型
                print(f"\n[{timestamp}] [Unknown]: {evt}")
                
        except json.JSONDecodeError:
            # 非JSON消息（原始数据）
            print(f"\n[Raw]: {message}")

async def main():
    """
    主函数：解析命令行参数，连接WebSocket服务器，并发执行发送和接收任务
    """
    parser = argparse.ArgumentParser(description="Qwen3-ASR Streaming Client")
    parser.add_argument("-e", "--endpoint", required=True, help="WebSocket Endpoint URL (e.g. ws://localhost:8907/transcribe-streaming)")
    parser.add_argument("-f", "--file", required=True, help="Path to raw PCM 16k 16-bit mono file (or WAV with correct format)")
    args = parser.parse_args()

    print(f"Connecting to {args.endpoint}...")
    
    # 计算音频文件时长
    file_size = os.path.getsize(args.file)  # 文件大小（字节）
    duration = file_size / 32000.0  # 时长 = 文件大小 / (采样率 * 每样本字节数) = 文件大小 / (16000 * 2)
    print(f"Audio Duration: {duration:.2f}s")
    
    start_time = time.time()
    try:
        # 连接到WebSocket服务器（max_size=None表示不限制消息大小）
        async with websockets.connect(args.endpoint, max_size=None) as ws:
            # 并发执行发送和接收任务
            await asyncio.gather(sender(ws, args.file), receiver(ws))
            
        # 计算处理时间和实时因子（RTF）
        end_time = time.time()
        process_time = end_time - start_time  # 总处理时间
        rtf = process_time / duration  # 实时因子：处理时间 / 音频时长（<1表示快于实时，>1表示慢于实时）
        print(f"\nProcessing Time: {process_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.4f}")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
