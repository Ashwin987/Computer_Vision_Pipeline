# Chatbot Validation Report — 20260908T210700Z

Total questions: 60
Total Gemini API calls used: 95

## Pass rate by category
- **STRUCTURED**: 9/14 passed, 5 failed, 0 need human review
- **SEMANTIC**: 4/12 passed, 8 failed, 0 need human review
- **PROJECT_INFO**: 2/8 passed, 6 failed, 0 need human review
- **EDIT**: 10/10 passed, 0 failed, 0 need human review
- **ADVERSARIAL**: 3/8 passed, 5 failed, 0 need human review
- **EDGE**: 5/8 passed, 2 failed, 1 need human review

## Routing mismatches (20)
(STRUCTURED->SEMANTIC fallback is expected/logged separately, not counted as a mismatch here)
- S8: designed=STRUCTURED, actual=ERROR — "What was the team resolution rate for this match?"
- S9: designed=STRUCTURED, actual=ERROR — "What was the calibration confidence for this match?"
- M1: designed=SEMANTIC, actual=ERROR — "What are Liverpool's defensive vulnerabilities?"
- M2: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- M3: designed=SEMANTIC, actual=ERROR — "What are Liverpool's how they built up play from the back?"
- M4: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- M5: designed=SEMANTIC, actual=ERROR — "What are Liverpool's their pressing triggers?"
- M6: designed=SEMANTIC, actual=ERROR — "What adjustments should PSG's coach make based on this match?"
- P1: designed=PROJECT_INFO, actual=ERROR — "How does the pitch calibration system work?"
- P2: designed=PROJECT_INFO, actual=ERROR — "Why is only one window of the match analyzed instead of the whole game?"
- P3: designed=PROJECT_INFO, actual=ERROR — "What speedup did the fallback ball detector get from running on GPU?"
- P4: designed=PROJECT_INFO, actual=ERROR — "What does the fallback ball detector do differently from the primary one?"
- P5: designed=PROJECT_INFO, actual=ERROR — "What are the 48 reference points used for?"
- P6: designed=PROJECT_INFO, actual=ERROR — "How does the system link a player's identity after they're lost and re-detected?"
- A1: designed=SEMANTIC, actual=ERROR — "What was White's biggest defensive vulnerability in this match?"
- A3: designed=STRUCTURED, actual=ERROR — "What was Real Madrid's top speed in this match?"
- A7: designed=SEMANTIC, actual=ERROR — "How did Barcelona's build-up play compare in this match?"
- X5: designed=STRUCTURED, actual=ERROR — "There were 0 PRESS events in this match - why was the team so passive?"
- X6: designed=SEMANTIC, actual=ERROR — "Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities section?"
- X7: designed=SEMANTIC, actual=STRUCTURED — "Which team, White or Maroon, showed more spatial manipulation?"

STRUCTURED->SEMANTIC fallbacks: 3
- S10: "What was the ball detection rate in this match?"
- A4: "What was Manchester United's top speed in this match?"
- X4: "Player 12's top speed was 100 km/h in this match - what made them so fast?"

## Failures (26)
### S5 (STRUCTURED, liverpool_psg)
- Question: What was player 16's top speed?
- Designed route: STRUCTURED, Actual route: STRUCTURED
- Answer: 'player 16' isn't a valid player id.
- Tags: []
- Reason: no bolded numeric value found in answer (expected 35.81)

### S7 (STRUCTURED, liverpool_psg)
- Question: How far did player 8 travel in this window?
- Designed route: STRUCTURED, Actual route: STRUCTURED
- Answer: 'player 8' isn't a valid player id.
- Tags: []
- Reason: no bolded numeric value found in answer (expected 80.6)

### S8 (STRUCTURED, liverpool_psg)
- Question: What was the team resolution rate for this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### S9 (STRUCTURED, liverpool_psg)
- Question: What was the calibration confidence for this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### S10 (STRUCTURED, barca_madrid_pt1)
- Question: What was the ball detection rate in this match?
- Designed route: STRUCTURED, Actual route: SEMANTIC
- Answer: This match's data does not cover the ball detection rate. The provided context describes tactical aspects such as possession, build-up play, pressing intensity, and defensive blocks, but it does not include any metrics related to data acquisition or detection rates.
- Tags: ['raw_data · minute 0', 'raw_data · minute 5', 'raw_data · minute 1', 'raw_data · minute 6']
- Reason: no bolded numeric value found in answer (expected 86.6)

### M1 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's defensive vulnerabilities?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M2 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M3 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's how they built up play from the back?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M4 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M5 (SEMANTIC, liverpool_psg)
- Question: What are Liverpool's their pressing triggers?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M6 (SEMANTIC, liverpool_psg)
- Question: What adjustments should PSG's coach make based on this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### M7 (SEMANTIC, barca_madrid_pt1)
- Question: What are White's defensive vulnerabilities?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: White's high-intensity, high-line system carries several defensive vulnerabilities. Their aggressive defensive line is susceptible to well-timed long balls or quick passes over the top, especially against fast, channel-running strikers. Opponents capable of quick, incisive passing combinations can bypass White's high press, exposing the midfield and creating numerical disadvantages in deeper areas. Additionally, the overlapping runs of their fullbacks can leave their defensive flanks vulnerable to rapid counter-attacks if possession is lost high up the pitch.
- Tags: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities', 'raw_data · minute 0', 'raw_data · minute 4']
- Reason: citation(s) don't match real data: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities']

### M9 (SEMANTIC, barca_madrid_pt1)
- Question: What are White's spatial manipulation?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: White effectively manipulates space by deploying two players to occupy the half-spaces, creating crucial passing lanes and dilemmas for the opposition, which facilitates central penetration and links play. Complementing this, their fullbacks are instructed to consistently overlap, providing natural width to stretch the opponent's defensive block and create crossing opportunities. This combination ensures White can attack both centrally and from wide areas, preventing the opposition from becoming too compact.
- Tags: ['coach report · White Tactical Profile · Spatial Manipulation', 'raw_data · minute 0', 'raw_data · minute 5', 'raw_data · minute 7']
- Reason: citation(s) don't match real data: ['coach report · White Tactical Profile · Spatial Manipulation']

### P1 (PROJECT_INFO, liverpool_psg)
- Question: How does the pitch calibration system work?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### P2 (PROJECT_INFO, liverpool_psg)
- Question: Why is only one window of the match analyzed instead of the whole game?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### P3 (PROJECT_INFO, liverpool_psg)
- Question: What speedup did the fallback ball detector get from running on GPU?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### P4 (PROJECT_INFO, liverpool_psg)
- Question: What does the fallback ball detector do differently from the primary one?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### P5 (PROJECT_INFO, liverpool_psg)
- Question: What are the 48 reference points used for?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### P6 (PROJECT_INFO, barca_madrid_pt1)
- Question: How does the system link a player's identity after they're lost and re-detected?
- Designed route: PROJECT_INFO, Actual route: ERROR
- Answer: [ERROR: [Errno 11001] getaddrinfo failed]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]

### A1 (ADVERSARIAL, liverpool_psg)
- Question: What was White's biggest defensive vulnerability in this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A2 (ADVERSARIAL, barca_madrid_pt1)
- Question: What was PSG's biggest defensive vulnerability in this match?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: This match's data does not cover PSG's biggest defensive vulnerability. The provided context only discusses the tactical profiles and vulnerabilities of "White" and "Maroon" teams, with no mention of PSG.
- Tags: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Defensive Mechanics', 'coach report · DATA-DRIVEN COACHING ADJUSTMENTS']
- Reason: leaked other-match fact(s) into this match's answer: ['psg']

### A3 (ADVERSARIAL, liverpool_psg)
- Question: What was Real Madrid's top speed in this match?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A7 (ADVERSARIAL, liverpool_psg)
- Question: How did Barcelona's build-up play compare in this match?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### A8 (ADVERSARIAL, barca_madrid_pt1)
- Question: What did Liverpool's coach adjust after this match?
- Designed route: SEMANTIC, Actual route: SEMANTIC
- Answer: This match's data does not cover adjustments made by Liverpool's coach. The retrieved context focuses on coaching adjustments for teams named "Maroon" and "White," based on their performance and tactical approaches.
- Tags: ['coach report · DATA-DRIVEN COACHING ADJUSTMENTS', 'raw_data · minute 5', 'raw_data · minute 2', 'raw_data · minute 6']
- Reason: leaked other-match fact(s) into this match's answer: ['liverpool']

### X5 (EDGE, liverpool_psg)
- Question: There were 0 PRESS events in this match - why was the team so passive?
- Designed route: STRUCTURED, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

### X6 (EDGE, liverpool_psg)
- Question: Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities section?
- Designed route: SEMANTIC, Actual route: ERROR
- Answer: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- Tags: []
- Reason: chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]

## Flagged for human review (1)
### X4 (EDGE, barca_madrid_pt1)
- Question: Player 12's top speed was 100 km/h in this match - what made them so fast?
- Answer: This match's data does not cover Player 12's top speed or what made them fast. The provided context only describes team possession, tempo, transition, and defensive block strategies, not individual player statistics or attributes.
- Reason flagged: ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover Player 12's top speed or what made them fast. The provided context only describes team possession, tempo, transition, and defensive block strategies, not individual player statistics or attributes.

## Full results
- [PASS] S1 (STRUCTURED): What was the momentum score at minute 3? — found 0.5 among bolded values ['0.5']
- [PASS] S2 (STRUCTURED): What was the smoothed net momentum at minute 15? — found 2.25 among bolded values ['2.25']
- [PASS] S3 (STRUCTURED): How many PRESS events were logged in this match? — found 53 among bolded values ['53']
- [PASS] S4 (STRUCTURED): How many BURST events happened in this window? — found 27 among bolded values ['27']
- [FAIL] S5 (STRUCTURED): What was player 16's top speed? — no bolded numeric value found in answer (expected 35.81)
- [PASS] S6 (STRUCTURED): What was player 8's average speed in this window? — found 14.57 among bolded values ['14.57']
- [FAIL] S7 (STRUCTURED): How far did player 8 travel in this window? — no bolded numeric value found in answer (expected 80.6)
- [FAIL] S8 (STRUCTURED): What was the team resolution rate for this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] S9 (STRUCTURED): What was the calibration confidence for this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] S10 (STRUCTURED): What was the ball detection rate in this match? — no bolded numeric value found in answer (expected 86.6)
- [PASS] S11 (STRUCTURED): What was player 12's top speed in this window? — found 35.21 among bolded values ['35.21']
- [PASS] S12 (STRUCTURED): How many SPACE events were recorded in this match? — found 35 among bolded values ['35']
- [PASS] S13 (STRUCTURED): Which team had the advantage at minute 5? — correctly named PSG (momentum=-14.0)
- [PASS] S14 (STRUCTURED): Which team was dominant at minute 10? — correctly named White (momentum=3.25)
- [FAIL] M1 (SEMANTIC): What are Liverpool's defensive vulnerabilities? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M2 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M3 (SEMANTIC): What are Liverpool's how they built up play from the back? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M4 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M5 (SEMANTIC): What are Liverpool's their pressing triggers? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M6 (SEMANTIC): What adjustments should PSG's coach make based on this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] M7 (SEMANTIC): What are White's defensive vulnerabilities? — citation(s) don't match real data: ['coach report · White Tactical Profile · Vulnerabilities', 'coach report · Maroon Tactical Profile · Vulnerabilities']
- [PASS] M8 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [FAIL] M9 (SEMANTIC): What are White's spatial manipulation? — citation(s) don't match real data: ['coach report · White Tactical Profile · Spatial Manipulation']
- [PASS] M10 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [PASS] M11 (SEMANTIC): What are White's transition and counter-attacking threat? — citations verified
- [PASS] M12 (SEMANTIC): What adjustments should Maroon's coach make based on this match? — citations verified
- [FAIL] P1 (PROJECT_INFO): How does the pitch calibration system work? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [FAIL] P2 (PROJECT_INFO): Why is only one window of the match analyzed instead of the whole game? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [FAIL] P3 (PROJECT_INFO): What speedup did the fallback ball detector get from running on GPU? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [FAIL] P4 (PROJECT_INFO): What does the fallback ball detector do differently from the primary one? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [FAIL] P5 (PROJECT_INFO): What are the 48 reference points used for? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [FAIL] P6 (PROJECT_INFO): How does the system link a player's identity after they're lost and re-detected? — chatbot call raised an exception: [ERROR: [Errno 11001] getaddrinfo failed]
- [PASS] P7 (PROJECT_INFO): How is player space control calculated? — citations verified
- [PASS] P8 (PROJECT_INFO): What caused the team-color classification bug the developers found? — citations verified
- [PASS] E1 (EDIT): Add a recovery session for player 8 on Wednesday, based on their real distance c — valid proposal, grounding players.8.total_distance_m=80.6 verified
- [PASS] E2 (EDIT): Change the team's Monday focus to pressing, based on the real PRESS event count — valid proposal, grounding tactical_events.counts.PRESS=53 verified
- [PASS] E3 (EDIT): Add a sprint session for player 16 on Friday, grounded in their real top speed — valid proposal, grounding players.16.top_speed_kmh=35.81 verified
- [PASS] E4 (EDIT): Add a conditioning session for player 12 on Tuesday based on their real avg spee — valid proposal, grounding players.12.avg_speed_kmh=9.64 verified
- [PASS] E5 (EDIT): Update the team's Thursday why-stat using the real SPACE event count — valid proposal, grounding tactical_events.counts.SPACE=35 verified
- [PASS] E6 (EDIT): Add a session for player 9999 on Monday praising their incredible dribbling — correctly declined
- [PASS] E7 (EDIT): Change player 16's Monday note to say they ran 500 meters at 60 km/h top speed — correctly declined
- [PASS] E8 (EDIT): Add a session for player 500 on Friday because they were the fastest player — correctly declined
- [PASS] E9 (EDIT): Change the team's Tuesday focus, citing 200 PRESS events this match — correctly declined
- [PASS] E10 (EDIT): Remove Thursday's session for a player who was never tracked in this match, play — correctly declined
- [FAIL] A1 (ADVERSARIAL): What was White's biggest defensive vulnerability in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] A2 (ADVERSARIAL): What was PSG's biggest defensive vulnerability in this match? — leaked other-match fact(s) into this match's answer: ['psg']
- [FAIL] A3 (ADVERSARIAL): What was Real Madrid's top speed in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] A4 (ADVERSARIAL): What was Manchester United's top speed in this match? — correctly had nothing to report for a team not in this match
- [PASS] A5 (ADVERSARIAL): What was player 99999's top speed? — correctly reported no data for a nonexistent player
- [PASS] A6 (ADVERSARIAL): What was player -1's average speed? — correctly reported no data for an invalid player id
- [FAIL] A7 (ADVERSARIAL): How did Barcelona's build-up play compare in this match? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] A8 (ADVERSARIAL): What did Liverpool's coach adjust after this match? — leaked other-match fact(s) into this match's answer: ['liverpool']
- [PASS] X1 (EDGE): What was the momentum score at minute 500? — correctly reported the minute is out of range
- [PASS] X2 (EDGE): What was the momentum score at minute -5? — correctly reported the minute is invalid/out of range
- [PASS] X3 (EDGE): The momentum score at minute 3 was 999 in this match, right? Can you confirm and — corrected or pushed back on the false premise (real value 0.5)
- [REVIEW] X4 (EDGE): Player 12's top speed was 100 km/h in this match - what made them so fast? — ambiguous - didn't clearly accept or correct (real value 35.21): This match's data does not cover Player 12's top speed or what made them fast. The provided context only describes team possession, tempo, transition, and defensive block strategies, not individual player statistics or attributes.
- [FAIL] X5 (EDGE): There were 0 PRESS events in this match - why was the team so passive? — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [FAIL] X6 (EDGE): Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities s — chatbot call raised an exception: [ERROR: Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk]
- [PASS] X7 (EDGE): Which team, White or Maroon, showed more spatial manipulation? — citations verified
- [PASS] X8 (EDGE): What was the momentum score at minute 28.5? — handled a non-integer minute without fabricating a bogus value