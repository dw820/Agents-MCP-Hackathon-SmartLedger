---
title: SmartLedger
emoji: 😻
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
tags:
  - mcp-server-track
  - agent-demo-track
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Agents-MCP-Hackathon-SmartLedger

[![Sync to HuggingFace Space](https://github.com/dw820/Agents-MCP-Hackathon-SmartLedger/actions/workflows/main.yml/badge.svg)](https://huggingface.co/spaces/Agents-MCP-Hackathon/SmartLedger)

## Demo Video
[link](https://x.com/WeiTu_/status/1931950511332417906)


## Project Context

SmartLedger is an intelligent financial reconciliation system designed to solve the complex challenges faced by small businesses in managing their accounting processes. The project was developed as part of the Agents-MCP-Hackathon, showcasing advanced AI capabilities in financial document processing and transaction matching.

## Problem Statement

Small businesses often struggle with time-consuming manual reconciliation processes that are prone to human error. Traditional accounting workflows require:

- **Manual Data Entry**: Tedious transcription of transactions from bank statements, receipts, and invoices
- **Time-Intensive Matching**: Hours spent cross-referencing transactions between different financial documents
- **Error-Prone Processes**: High risk of mistakes when manually comparing dates, amounts, and vendor information
- **Limited Analysis**: Difficulty in extracting meaningful insights from financial data without specialized expertise
- **Fragmented Systems**: Multiple tools and spreadsheets that don't communicate effectively

SmartLedger addresses these pain points by providing an AI-powered solution that:

- Automatically extracts transaction data from images using computer vision
- Intelligently matches transactions across different financial documents
- Provides confidence scoring for automated reconciliation decisions
- Offers natural language querying for financial insights
- Reduces manual effort while improving accuracy and compliance

The system is particularly valuable for small business owners, bookkeepers, and accountants who need efficient, reliable tools for financial reconciliation without the complexity or cost of enterprise-level solutions.

## Tech Stack

- **Gradio**: Frontend UI and MCP server for interactive web interface and programmatic access
- **Modal**: Serverless compute platform for scalable document processing functions
- **Hyperbolic**: Model inference platform powering Qwen2.5-VL-7B-Instruct model for AI-powered document parsing and transaction extraction from images
