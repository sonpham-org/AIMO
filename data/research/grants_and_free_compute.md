# Grants & Free Compute for AIMO3 — February 7, 2026

All grants and free compute resources relevant to our AIMO3 competition work (math reasoning, LLM fine-tuning, GPU compute). Sorted by relevance and likelihood of success.

---

## Tier 1: AIMO3-Specific (Apply ASAP)

### 1. Fields Model Initiative — Free 128x H100 GPUs
- **What**: Up to 128 H100 GPUs for fine-tuning models for AIMO3
- **Cost**: FREE
- **Partners**: National Institute of Informatics (Tokyo), Benchmarks+Baselines (Vienna)
- **Eligibility**: AIMO3 competition participants ("select participants, see Kaggle for criteria")
- **How to apply**: Through the Kaggle AIMO3 competition page
- **URL**: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/
- **Notes**: This is the single most valuable resource. 128 H100s would allow full GRPO RL training runs on gpt-oss-120b. Check Kaggle competition rules/discussion for specific application instructions.
- **Priority**: HIGHEST

### 2. Tinker Credits via AIMO3 Partnership
- **What**: Tinker API credits for AIMO3 participants
- **Cost**: FREE
- **How to apply**: Through the Kaggle AIMO3 competition page
- **Notes**: Tinker directly supports gpt-oss-120b fine-tuning (SFT + RL). Credits specifically for AIMO3 competitors.
- **Priority**: HIGH

---

## Tier 2: Research Grants (Apply This Week)

### 3. Tinker Research Grant — $5,000+ Credits
- **What**: Tinker API credits for fine-tuning open-weight LLMs
- **Amount**: Starting at $5,000 (could be more based on project)
- **Eligibility**: "Research projects and open-source software that uses Tinker"
- **Application**: Typeform at https://thinkingmachines.ai/blog/tinker-research-and-teaching-grants/
- **Timeline**: Rolling applications, ~1 week response time
- **Notes**: Very relevant to us. Pitch: "Fine-tuning gpt-oss-120b for mathematical reasoning using GRPO RL with tool-integrated reasoning." They specifically support RL training via their Math RL recipe in the tinker-cookbook.
- **Estimate**: $5K credits = ~125M tokens of gpt-oss-120b training = multiple full RL runs
- **Priority**: HIGH

### 4. Prime Intellect Fast Compute Grants — $500-$100,000
- **What**: Compute credits on Prime Intellect platform
- **Amount**: $500 to $100,000
- **Eligibility**: Anyone working on GPU-intensive open-source AI projects
- **How to apply**: Email pitch to contact@primeintellect.ai
- **Timeline**: 5-10 day response
- **Notes**: High bar for quality but no hoops. Pitch: AIMO3 competition + open-source math reasoning research. They value novel approaches.
- **URL**: https://www.primeintellect.ai/blog/fast-compute-grants
- **Priority**: HIGH

### 5. Lambda AI Research Credits — Up to $5,000
- **What**: Cloud credits on Lambda GPU cloud (H100s available)
- **Amount**: Up to $5,000
- **Eligibility**: AI researchers
- **How to apply**: https://lambda.ai/research#grant-application
- **Notes**: Lambda has H100s at competitive rates (~$2/hr). $5K = ~2,500 H100-hours, enough for significant RL training.
- **Priority**: HIGH

---

## Tier 3: General Research Programs (Apply When Ready)

### 6. Nebius Research Credits — Up to 8 GPUs for 1 Year
- **What**: GPU cloud access + 10M inference tokens
- **Amount**: Up to 8 GPUs for 12 months
- **Eligibility**: Researchers at accredited academic/nonprofit institutions
- **How to apply**: https://nebius.com/nebius-research-credits-program (monthly application windows, 2-week cycles)
- **Status**: Currently between cycles — watch for next opening
- **Notes**: Very generous if you qualify (academic affiliation helps)
- **Priority**: MEDIUM

### 7. Google Cloud Research Credits — Up to $5,000
- **What**: Google Cloud credits (includes GPU instances)
- **Amount**: Up to $5,000 (faculty/postdoc), $1,000 (PhD students)
- **Eligibility**: Faculty, postdoc, or PhD student at accredited institution
- **How to apply**: https://edu.google.com/programs/credits/research/
- **Timeline**: 6-8 weeks review
- **Notes**: One-time application for faculty. Can use for GPU instances but Google Cloud GPU pricing is higher than dedicated providers.
- **Priority**: MEDIUM

### 8. Together AI Research Credits
- **What**: Compute credits on Together AI platform
- **Eligibility**: Students conducting independent research
- **How to apply**: https://www.together.ai/forms/research-credits-program-request
- **Requirement**: Acknowledge Together AI in published work
- **Priority**: MEDIUM

### 9. AWS Cloud Credit for Research — Up to $5,000+
- **What**: AWS cloud credits
- **Amount**: Up to $5,000 (students), uncapped (faculty/staff)
- **How to apply**: https://pages.awscloud.com/aws-cloud-credit-for-research.html
- **Contact**: aws-research-credit@amazon.com
- **Notes**: Check spam for responses
- **Priority**: MEDIUM

### 10. Fal AI Research Grants
- **What**: Free compute for open-source AI projects
- **Eligibility**: Anyone passionate about AI + open source. No degree required.
- **How to apply**: Email [email protected] with project description
- **Priority**: LOW-MEDIUM

### 11. HOSTKEY GPU Grant Program
- **What**: GPU resources for scientific projects
- **Eligibility**: Data science professionals/researchers
- **How to apply**: https://landing.hostkey.com/grant_for_scientific_projects_formhtml
- **Priority**: LOW-MEDIUM

---

## Tier 4: Startup Programs (If Applicable)

These require being a startup/company but offer massive credits:

| Program | Amount | GPU Access | Eligibility |
|---------|--------|------------|-------------|
| Google for Startups (AI) | $350,000 | H100, TPU v5e | VC-backed AI startup, up to Series A |
| AWS Activate | $100K-$300K | H100 (P5) | VC intro, up to Series B |
| Microsoft for Startups | $150,000 | A100 instances | Product live + traction |
| DigitalOcean Hatch | $100,000 + 3mo H100 | 8x H100 droplets | Pre-seed to Series A |
| IBM Startup | $120,000 | L40S, A100 | <5 years old, <$1M revenue |
| NVIDIA Inception | $100K AWS credits | Preferred GPU pricing | Incorporated, active dev |

---

## Tier 5: Academic Fellowships (Long-term)

### NVIDIA Academic Grant Program
- **What**: Up to 30,000 H100 80GB hours + optional hardware
- **Eligibility**: Full-time faculty at PhD-granting institutions
- **Research areas**: GenAI, LLMs, simulation, scientific computing
- **Timeline**: Quarterly cycles (Jan-Mar submit -> Jun decision)
- **URL**: https://academicgrants.nvidia.com/academicgrantprogram/s/Application
- **Notes**: Requires using NVIDIA software. 30K H100 hours = massive RL training budget.

### NVIDIA Graduate Fellowship
- **What**: $60,000 stipend for PhD students
- **Eligibility**: PhD students in CS/AI/EE, past first year
- **Deadline**: September 15 annually (2025 cycle already closed)
- **Notes**: Next cycle opens ~Aug 2026

### AI for Math Fund (XTX Markets / Renaissance Philanthropy)
- **What**: Part of $9M fund for AI+math projects
- **Focus**: Formal verification, theorem proving, open-source math tools
- **Eligibility**: Academic, commercial, or independent projects
- **Contact**: aiformath@renphil.org
- **Notes**: More focused on formal math than competition math, but worth exploring
- **URL**: http://www.renaissancephilanthropy.org/ai-for-math-fund/

### Hugging Face ZeroGPU + Community Grants
- **What**: Free H200 GPU access for Spaces demos; community GPU grants for Spaces
- **How**: Deploy model on HF Spaces, apply for community GPU grant in Settings tab
- **Notes**: Good for inference/demos, not training. ZeroGPU is shared/dynamic.

---

## Recommended Application Order

**This week (February 7-14):**
1. Check Kaggle AIMO3 page for Fields Model Initiative application (128 H100s free)
2. Check Kaggle AIMO3 page for Tinker credits partnership
3. Apply for Tinker Research Grant ($5K+) via Typeform
4. Email Prime Intellect (contact@primeintellect.ai) for Fast Compute Grant
5. Apply for Lambda Research Credits ($5K)

**Next week (February 14-21):**
6. Apply for Together AI Research Credits
7. Apply for Google Cloud Research Credits ($5K)
8. Email Fal AI ([email protected]) for research grant
9. Check Nebius for next application window

**Pitch template for applications:**
> We are competing in AIMO3 (AI Mathematical Olympiad Progress Prize 3, $2.2M prize pool on Kaggle). We are developing improved answer selection strategies and fine-tuning the gpt-oss-120b model for mathematical reasoning using tool-integrated reasoning (TIR) and reinforcement learning. Our current approach scores 40/50 on the public leaderboard. We plan to open-source our methods and findings.

---

## Total Potential Value

| Source | Potential Value | Likelihood |
|--------|----------------|------------|
| Fields Model Initiative | 128x H100 GPUs (worth $50K+) | Medium-High |
| Tinker AIMO3 credits | $1K-5K | High |
| Tinker Research Grant | $5K-10K | Medium-High |
| Prime Intellect | $500-$10K | Medium |
| Lambda Research | $5K | Medium |
| Google Cloud | $1K-5K | Medium |
| Together AI | $500-2K | Medium |
| AWS Research | $5K+ | Medium |

**Conservative estimate**: $5K-15K in free compute
**Optimistic estimate**: $30K-70K+ (if Fields Initiative + multiple grants)
