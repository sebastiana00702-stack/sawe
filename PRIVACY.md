# Privacy Policy for Sawe

**Last updated:** May 18, 2026

## Overview

Sawe is a personal, non-commercial running coach project that uses fitness 
and recovery data from the WHOOP wearable to generate individualized workout 
recommendations. This privacy policy explains what data Sawe accesses, how 
it is used, and how it is stored.

## Who operates Sawe

Sawe is an open-source personal project developed and operated by Sebastian 
Alvarez for individual use. It is not a company, not a product offered for 
sale, and not affiliated with WHOOP, Inc. The source code is available at 
https://github.com/sebastiana00702-stack/sawe.

## What data Sawe accesses

When you authorize Sawe via WHOOP's OAuth, Sawe requests permission to read 
the following from your WHOOP account:

- Recovery scores (daily 0–100 score, HRV, resting heart rate, respiratory 
  rate, skin temperature)
- Sleep records (duration, performance, stages)
- Cycle (daily strain) data
- Workout records (heart rate, strain, duration)
- Basic profile information

Sawe does not access, request, or store financial information, social 
contacts, location data, or any data not explicitly listed above.

## How data is used

WHOOP data is used solely to:

- Compute derived metrics (rolling baselines, acute and chronic training 
  load, training monotony)
- Evaluate evidence-based training and recovery rules
- Generate daily workout recommendations specific to the user

Data is never sold, shared with advertisers, used to train machine learning 
models, or transmitted to any third party.

## Where data is stored

Sawe runs locally on the user's own machine. Data fetched from the WHOOP API 
is cached on the local device only. There is no Sawe-operated server, cloud 
database, or remote storage. No data leaves the user's device except for the 
original API calls to WHOOP.

OAuth credentials (refresh tokens) are stored in a local environment file 
(`.env`) on the user's machine and are not transmitted elsewhere.

## Data retention and deletion

Users may delete all Sawe-cached data at any time by deleting the project 
directory or the local cache file (`data/whoop_cache.json`). Revoking Sawe's 
authorization in the WHOOP developer dashboard immediately terminates Sawe's 
ability to fetch new data.

## Security

Because Sawe runs locally, security depends on the security of the user's 
own device. The user is responsible for keeping their `.env` file and OAuth 
credentials confidential. Sawe does not transmit credentials anywhere except 
to WHOOP's own OAuth endpoints.

## Children's privacy

Sawe is not intended for use by individuals under 18.

## Medical disclaimer

Sawe is not a medical device and does not provide medical advice. 
Recommendations are for general fitness purposes only. Users should consult 
a qualified healthcare professional before beginning, modifying, or 
stopping any exercise program. If Sawe's output suggests potential medical 
concerns (illness flags, persistent overreaching), users should consult a 
clinician — Sawe is not a substitute for medical evaluation.

## Changes to this policy

Updates to this privacy policy will be committed to the project repository 
with a revised "Last updated" date.

## Contact

For questions about this privacy policy or Sawe's data handling, contact 
Sebastian Alvarez via the GitHub repository linked above.
