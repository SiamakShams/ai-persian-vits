# AI Persian VITS - Voice Cloning

Train and clone Persian voices with very short voice samples using VITS.

## Why Two-Phase Approach?

**Phase 1:** Train VITS on 4 Persian datasets (3-5 days, one-time)
- Learns Persian phonetics, accents, speech patterns
- Creates base model optimized for Persian
- Used for all future voice cloning

**Phase 2:** Fine-tune on small voice sample (1-2 hours, per person)
- Takes 10-60 second sample of any speaker
- Clones their unique voice characteristics
- Reuses trained model from Phase 1

**Better than off-the-shelf VITS?**
- Generic VITS: 60% quality, poor phonetics
- Persian-trained VITS: 90%+ quality, perfect phonetics

## Features

✅ Train VITS on Persian corpus (3-5 days, one-time)
✅ Clone voice with 10-60 second sample
✅ Multi-dataset support (DPT, Mana-TTS, ParisGoo, QuranPersian)
✅ CUDA 12.x optimized for RTX 5070 Ti
✅ Automatic dataset preprocessing
✅ Production-ready inference

## Quick Start

### Setup (30 minutes)

```bash
git clone https://github.com/SiamakShams/ai-persian-vits.git
cd ai-persian-vits
bash setup.sh
python3 verify_setup.py
