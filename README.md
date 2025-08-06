# 🚀 AI & Tech Jobs Dataset

**Daily-updated dataset tracking 932+ engineering positions across 114+ top AI/tech companies**

[![Update Status](https://img.shields.io/badge/Updated-Daily-green)]()
[![Companies](https://img.shields.io/badge/Companies-114+-orange)]()
[![Positions](https://img.shields.io/badge/Positions-932+-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 🎯 What's This?

An open-source dataset of AI and tech job opportunities, automatically updated daily. Perfect for:

- 💼 **Job Seekers**: Find opportunities at top AI companies
- 📊 **Researchers**: Analyze hiring trends in AI/tech
- 🤖 **Developers**: Build job-related applications
- 📈 **Market Analysis**: Track talent demand

## 🏢 Featured Companies

### AI Foundation Models (3 companies)
OpenAI, Anthropic, xAI

### AI Developer Tools (6 companies) 
Cursor, Replit, Mintlify, Sourcegraph, Sentry, Stainless API

### AI Data & Analytics (6 companies)
Databricks, MongoDB, AlphaSense, Labelbox, Chalk, Tonic AI

### AI Security & Enterprise (5 companies)
Abnormal Security, Vanta, Semgrep, OSO, Kasada

### AI Hardware & Robotics (5 companies)
Anduril, Skydio, Gecko Robotics, SandboxAQ, Shield AI

### AI FinTech (4 companies)
Addepar, Ridgeline, Imprint, Extend

### AI HealthTech (4 companies)
Abridge, Anterior, Tennr, Flagship Pioneering

*And 81+ other innovative AI & tech companies...*


* If you would like to see final results in Notion:
https://www.notion.so/xiaoshuaifm/22aa175fa91c80c1a09fe1911721e2cc?v=22aa175fa91c80439bd0000c8ddbdf1f

## 📊 Quick Stats

- **932** active positions
- **114** companies tracked
- **13** role types indexed
- **64** locations covered
- **Daily** updates

## 🗂️ Data Structure

```
data/
├── companies/           # Individual company files
│   ├── openai.json     # 67 positions
│   ├── anthropic.json  # 21 positions
│   ├── databricks.json # 68 positions
│   └── ...
└── indexes/            # 🚀 MCP-optimized search indexes
    ├── master.json     # Company overview & statistics
    ├── by_role.json    # Complete positions by role (725KB)
    └── by_location.json # Complete positions by location (516KB)
```

### 🎯 **For MCP Users: Use Index Files!**
**Recommended approach**: Use `by_role.json` or `by_location.json` for efficient searches. Each index contains complete position data, not just IDs - perfect for single-request queries.

## 🚀 Quick Start

### Browse Companies
```bash
# List all companies
cat data/indexes/master.json | jq '.companies[].name'

# Get OpenAI jobs
cat data/companies/openai.json | jq '.positions[]'
```

### 🎯 Search by Role (MCP-Optimized)
```bash
# Get ALL ML Engineer positions with complete data (1 request!)
cat data/indexes/by_role.json | jq '.roles["machine-learning-engineer"][]'

# View role statistics first
cat data/indexes/by_role.json | jq '.role_statistics'

# Get Solutions Engineer roles
cat data/indexes/by_role.json | jq '.roles["solutions-engineer"][]'
```

### 🌍 Search by Location (MCP-Optimized)  
```bash
# Get ALL Remote USA positions with complete data (1 request!)
cat data/indexes/by_location.json | jq '.locations["Remote - USA"][]'

# View location statistics first
cat data/indexes/by_location.json | jq '.location_statistics'

# Get San Francisco positions  
cat data/indexes/by_location.json | jq '.locations["San Francisco, CA"][]'
```

## 📋 Position Data Format

Each position includes:
```json
{
  "role_name": "Forward Deployed Engineer",
  "location": "San Francisco, CA",
  "job_link": "https://jobs.ashbyhq.com/openai/...",
  "employment_type": "FullTime",
  "team": "Customer Success",
  "published_date": "2025-08-01",
  "job_id": "abc123",
  "first_seen": "2025-08-03",
  "last_seen": "2025-08-06",
  "status": "active"
}
```

## 🔄 Updates

- **Frequency**: Daily at 10:00 UTC
- **Source**: Public job postings
- **Quality**: Cleaned, deduplicated, and structured
- **History**: Track position lifecycle (new/updated/closed)

## 🛠️ For Developers

### 🚀 MCP-Optimized Access via HTTP
```bash
# Get complete ML Engineer positions (1 request!)
curl https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/indexes/by_role.json | jq '.roles["machine-learning-engineer"]'

# Get complete Remote positions (1 request!)  
curl https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/indexes/by_location.json | jq '.locations["Remote - USA"]'

# Traditional: Specific company
curl https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/companies/openai.json
```

### Python Usage (MCP-Optimized)
```python
import requests
import json

# MCP-friendly: Get ALL ML positions in 1 request
response = requests.get('https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/indexes/by_role.json')
ml_jobs = response.json()['roles']['machine-learning-engineer']

# MCP-friendly: Get ALL Remote positions in 1 request  
response = requests.get('https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/indexes/by_location.json')
remote_jobs = response.json()['locations']['Remote - USA']

# Traditional: Company overview
response = requests.get('https://raw.githubusercontent.com/ilovemyapps/fast-ai-startup-jobs-dataset/main/data/indexes/master.json')
companies = response.json()['companies']
```

## 📈 Trending Roles

Based on current data:
- **Software Engineer**: Most common role
- **Machine Learning Engineer**: High demand
- **Forward Deployed Engineer**: Specialized customer-facing roles
- **Solutions Engineer**: Growing category
- **Data Engineer**: Infrastructure focus

## 🏆 Top Hiring Companies

1. **Anduril** - 140 positions
2. **Databricks** - 68 positions  
3. **OpenAI** - 67 positions
4. **xAI** - 57 positions
5. **Pear VC Portfolio** - 36 positions

## 📝 License

MIT License - Free for commercial and research use.

## 🤝 Contributing

- Found a missing company? Open an issue!
- Data quality improvements? Submit a PR!
- Want to add more platforms? Let's discuss!

## ⚠️ Disclaimer

This dataset contains publicly available job posting information. All job listings link directly to official company career pages. Data is collected respectfully with appropriate delays and caching.

---

**Built with ❤️ for the AI/tech community** • Last updated: 2025-08-06