---
name: influencer-vetting
description: Checklist for vetting influencers for a brand campaign, including fake-follower red flags and content safety. Use whenever recommending or ranking influencers.
---

# Influencer vetting

For every candidate, call `get_influencer_metrics` and check:

1. **Fake-follower red flags**: flag the account and do NOT recommend it if any apply:
   - engagement rate below 1% with more than 500k followers
   - 30-day follower growth above 20% (bought followers)
   - audience country very different from the brand's target market
2. **Brand fit**: if a brand is named, call `get_brand_guidelines` and check engagement
   against `min_engagement_rate`, audience against `target_audience`, and past brands
   against `exclusions`.
3. **Content safety**: delegate each shortlisted handle to the `safety-reviewer` subagent
   (with the brand). Do NOT recommend UNSAFE accounts. Mention CAUTION items (e.g. an
   undisclosed sponsored post) next to the recommendation so the brand can decide.
4. **Output format**: a ranked list. Each line gives the handle, a one-line reason and
   the key numbers (followers, engagement %). Then add a "Flagged" section for any
   rejected accounts, with the reason.

Only use handles returned by the tools. Never invent influencers or numbers.
