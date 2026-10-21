"""
Dataset Curation & Generation Script
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector

This script compiles a representative benchmark dataset specifically designed
for WhatsApp forwards, viral rumors, phishing scams, and verified news articles.
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Curated WhatsApp Forwards (Suspicious / Misinformation = 1)
SUSPICIOUS_MESSAGES = [
    # Health & Medical Hoaxes
    "URGENT ALERT: WHO and UNESCO have declared that drinking hot water with lemon and baking soda kills 100% of coronavirus and all viral flu immediately! Doctor from AIIMS confirms! Share with every family group before midnight to save lives!!!",
    "BREAKING MEDICAL DISCOVERY: Bitter gourd juice mixed with raw turmeric can cure stage 4 cancer in just 7 days! Big pharma companies are hiding this cheap secret to sell expensive chemotherapy drugs. Please forward this message to everyone you love!",
    "WARNING FROM MINISTRY OF HEALTH: Avoid eating broiler chicken for the next 30 days as a deadly new virus has infected poultry farms across the state! Over 500 people hospitalized already. Do not ignore! Forward to all your WhatsApp contacts immediately!!",
    "Forwarded as received: If someone gets a heart attack, give them two pinches of cayenne pepper powder under their tongue within 60 seconds and they will recover instantly! No need to call ambulance! Confirmed by German cardiologists.",
    "ATTENTION: Microwaving plastic food containers creates toxic dioxins that cause instant leukemia and brain tumors. Japanese scientists have released an urgent red alert. Throw away all plastic containers immediately and forward this warning!",
    "Miracle cure revealed: Drinking pure cow urine (Gaumutra) at 5 AM on an empty stomach cures diabetes, high BP, and kidney stones permanently. Verified by international natural medicine council. Spread this wisdom to all groups!",
    "Alert: The new 5G towers being installed in cities are emitting dangerous electromagnetic radiation that causes sudden breathlessness and oxygen depletion in lungs. Keep your windows closed at night and switch off WiFi routers!",

    # Phishing & Financial Scams
    "Dear customer, your SBI / HDFC bank account will be blocked within 24 hours due to non-updation of PAN and KYC documents. Click here immediately to update your details and avoid Rs 10,000 fine: http://bit.ly/sbi-kyc-verify-urgent",
    "CONGRATULATIONS! Your mobile number has been selected in the WhatsApp Annual Lucky Draw 2026. You have won a cash prize of Rs 25,00,000! Contact our Mumbai head office claim officer Mr. Rajesh Sharma on WhatsApp: +91-9876543210 to claim your reward.",
    "Government of India has announced Free Laptop and Free Smartphone scheme for all college students under PM Yuva Digital Yojna 2026! Limited slots available. Register your Aadhaar and bank details now before 12 PM tonight: http://pmyuva-freelaptop.xyz",
    "Electricity department urgent notice: Your electricity power supply will be disconnected tonight at 9:30 PM from the power station because your last month electricity bill was not updated. Immediately call electricity officer at 9821034912.",
    "Exclusive Offer: Reliance Jio is celebrating its 10th anniversary by giving 1 Year Free 5G Recharge (3GB/day) to all users! Recharge is completely free for next 2 hours only. Click the link and forward to 10 friends to activate: http://jio-recharge.online",
    "Income Tax Department notification: An urgent income tax refund of Rs 15,480 has been approved for your account. Please confirm your netbanking username and debit card CVV to receive the direct credit into your account: http://incometax-refunds-portal.com",
    "Earn Rs 3000 to Rs 8000 daily working from home just by liking YouTube videos and rating Google Maps restaurants! No experience needed, instant daily payout via UPI. Message our HR manager on Telegram @EarnDailyFast.",

    # Fabricated Authority & False Claims
    "PROUD MOMENT FOR INDIA: UNESCO has just officially declared Jana Gana Mana as the BEST NATIONAL ANTHEM in the entire world! No other country's anthem matched its scientific composition and lyrical depth. Every true Indian must forward this to 10 groups right now! Jai Hind!",
    "NASA has just captured a live satellite image of India shining brightly on Diwali night! The photograph shows our entire nation illuminated in divine golden light. UNESCO also appreciated Indian culture. Forward this to make every Indian proud!",
    "URGENT NOTIFICATION FROM CENTRAL GOVERNMENT: Starting from Monday, all Rs 500 currency notes will be demonetized and replaced by new digital currency notes with embedded GPS microchips to stop corruption. Spend all your 500 notes immediately!",
    "WhatsApp will no longer be free from next week! If you do not forward this message to at least 15 active contacts, your WhatsApp icon will turn blue and you will be charged Rs 499 per month. Forward now to keep your account free and active forever!",
    "Secret documents leaked from Pentagon: Earth will experience 3 days of total darkness starting on December 21 due to cosmic planetary alignment and solar eclipse. Stock up on candles, canned food, and stay indoors. Do not go outside!",
    "BREAKING: The Supreme Court has passed an emergency order banning all private school fees during examinations. If any school asks for fees, file a complaint directly to police commissioner. Forward to all parents and school groups!",
    "UNESCO has declared Sanskrit as the most computer-friendly and algorithmically flawless language for 6th generation supercomputers and Artificial Intelligence. NASA will code upcoming Mars rovers in Sanskrit syntax.",
    "Alert from Cyber Crime Cell: Do not answer calls from phone numbers starting with +375, +371, or +381. Answering will hack your SIM card, copy your phonebook, and drain your entire bank balance in 3 seconds. Share to protect family!",

    # Sensational Rumors & Conspiracy
    "ATTENTION CITIZENS: Cold drinks like Frooti, Maaza, and Slice have been contaminated with blood infected with HIV by an angry factory worker in Hyderabad! NDTV has confirmed this news. Do not drink packaged mango drinks for next 2 months!",
    "Forwarded as received: If an ATM asks you to enter your PIN backwards in an emergency robbery, the machine secretly alerts the nearest police station while dispensing cash. Spread this life-saving secret trick to all your friends!",
    "SHOCKING TRUTH: The barcodes on grocery items indicate poison levels! Barcodes starting with 890 have toxic preservatives, while 730 are organic. Check your food labels before eating and warn everyone in your family group!",
    "Government circular leaked: All petrol and diesel vehicles older than 5 years will be impounded by traffic police starting from next month without any compensation. Protests happening nationwide. Share before government removes this post!",
    "A new deadly species of spider called the 'Silverback Recluse' has been spotted in public restrooms across metro stations. Its bite causes flesh necrosis within 15 minutes. See photo and share this emergency warning with everyone!",
    "WARNING: Do not charge your smartphone overnight while sleeping next to your pillow! The lithium battery emits toxic gamma frequencies that cause permanent neural damage and sudden explosion. Pass this on to all loved ones!",
    "Breaking News: Renowned international astrologer predicts a massive 9.2 magnitude earthquake hitting northern India between 2 AM and 5 AM tonight. Evacuate multistory buildings and sleep in open grounds. Share urgently!",
    "Forward this message to 10 friends and your phone battery will automatically charge to 100% due to electromagnetic vibration resonance protocol introduced in Android 15. It really works, try it now!"
]

# Curated Reliable / Authentic News & Messages (Reliable = 0)
RELIABLE_MESSAGES = [
    # Science & Technology Press Releases
    "The Indian Space Research Organisation (ISRO) successfully launched the NVS-02 navigation satellite aboard the GSLV-F12 rocket from the Satish Dhawan Space Centre in Sriharikota on Wednesday morning, mission directors confirmed in an official statement.",
    "NASA's James Webb Space Telescope has detected traces of water vapor, sulfur dioxide, and carbon dioxide in the atmosphere of an exoplanet orbiting a sun-like star located 700 light-years away, according to research published in Nature Astronomy.",
    "Researchers at MIT and Harvard University have developed an energy-efficient neuromorphic computing chip capable of processing sensor data with 85% less power consumption compared to traditional GPU architectures, the institute announced on Tuesday.",
    "The Nobel Prize in Chemistry was awarded to three scientists for their groundbreaking work on the synthesis of quantum dots and nanoscale semiconductors that now illuminate modern LED displays and biomedical imaging equipment.",
    "The European Space Agency announced that the Euclid spacecraft has completed its initial commissioning phase and began delivering high-resolution wide-angle surveys of distant galaxies to map cosmic dark matter distribution.",
    "A team of aerospace engineers at the Indian Institute of Technology (IIT) Madras has filed a patent for a novel low-cost electric propulsion system intended for micro-satellite constellations in low-Earth orbit.",

    # Official Government & Regulatory Advisories
    "The Reserve Bank of India (RBI) issued a notification advising bank account holders to remain vigilant against phishing links and cautioned that the central bank never solicits personal banking credentials, passwords, or OTPs through SMS or phone calls.",
    "The Ministry of Health and Family Welfare reiterated that routine immunization schedules for infants remain operational across all primary health centers nationwide and urged parents to follow official pediatric guidelines.",
    "The Press Information Bureau (PIB) Fact Check unit issued a clarification debunking a viral social media message claiming that the central government has discontinued currency notes, confirming that no such demonetization directive has been issued.",
    "The Ministry of Railways notified that ticket booking for seasonal festival special trains will open on the IRCTC portal starting at 10:00 AM on Friday, with standard tatkal booking terms applicable as per extant passenger reservation guidelines.",
    "The Department of Telecommunications (DoT) launched the Sanchar Saathi portal enabling mobile subscribers to verify mobile connections registered in their name and report stolen or lost handsets for immediate IMEI blocking.",
    "The Unique Identification Authority of India (UIDAI) has extended the provision of free document update for Aadhaar through the myAadhaar portal until December 31, encouraging residents to keep demographic details updated.",
    "The Central Board of Direct Taxes (CBDT) announced that over 7.28 crore income tax returns were filed for Assessment Year 2024-25, marking an increase of 7.5% over the previous fiscal period.",

    # Verified Health & Medical Information
    "The World Health Organization (WHO) published an updated guideline on dietary sodium intake, recommending that adults consume less than 2,000 milligrams of sodium per day to lower the risk of cardiovascular diseases and stroke.",
    "A randomized clinical trial published in The Lancet medical journal concluded that regular moderate aerobic exercise of 150 minutes per week significantly reduces the incidence of metabolic syndrome in adults with prediabetes.",
    "The Indian Council of Medical Research (ICMR) completed phase-3 trials for a recombinant vaccine against dengue, with the regulatory review committee scheduled to examine efficacy and safety data next month.",
    "Epidemiologists from the Centers for Disease Control and Prevention (CDC) emphasized that hydration and standard antipyretic medication under clinical supervision remain the recommended course for managing seasonal viral fevers.",
    "A systematic review conducted by the Cochrane Collaboration found no clinical evidence supporting the use of unverified herbal remedies for treating chronic hypertension without medical supervision.",

    # Economy, Public Affairs & Verified Current News
    "India's gross domestic product (GDP) grew at an annual rate of 7.2% in the previous fiscal quarter, driven primarily by strong manufacturing output and robust capital expenditure, official data released by the National Statistical Office showed.",
    "The Supreme Court of India today reserved its judgment on a batch of petitions challenging the validity of regulations concerning digital news dissemination and privacy safeguards under IT laws.",
    "The Securities and Exchange Board of India (SEBI) announced a consultation paper proposing tighter disclosure norms for algorithmic trading entities and retail derivative market participants to enhance investor protection.",
    "State-owned oil marketing companies announced a reduction of Rs 2 per liter in retail prices of petrol and diesel across the country following a moderation in international crude oil prices.",
    "The Delhi Metro Rail Corporation (DMRC) commenced commercial operations on the newly extended metro line connecting Janakpuri West to RK Ashram, reducing travel time across the outer ring road corridor.",
    "The India Meteorological Department (IMD) issued a weather advisory predicting normal to above-normal monsoon rainfall across central and peninsular India over the upcoming fortnight based on prevailing weather models.",
    "Public sector banks recorded an aggregate net profit exceeding Rs 1.4 lakh crore during the fiscal year, buoyed by declining non-performing assets and healthy credit growth across retail and MSME segments."
]

def augment_data(base_suspicious, base_reliable, target_count=300):
    """
    Expands the benchmark set with realistic lexical variations common in WhatsApp forwards.
    """
    random.seed(42)
    
    # Variations for suspicious messages
    suspicious_headers = [
        "URGENT FORWARD: ",
        "Forwarded as received: ",
        "BREAKING NEWS ALERT: ",
        "SHARE WITH ALL GROUPS! ",
        "VERY IMPORTANT WARNING: ",
        "DO NOT IGNORE THIS MESSAGE! ",
        "SHOCKING REALITY EXPOSED: ",
        "PLEASE READ AND SHARE IMMEDIATELY: "
    ]
    
    suspicious_footers = [
        " Forward to at least 10 people to spread awareness!",
        " Please share this before the government deletes this video/post!",
        " Save your family and friends by forwarding this right now!!!",
        " Do not delete without sharing to all your active WhatsApp contacts!",
        " This is 100% verified by top scientists. Share widely!",
        " Send to every group you are in immediately."
    ]
    
    # Variations for reliable messages
    reliable_prefixes = [
        "According to a press statement released on Tuesday, ",
        "In an official briefing today, ",
        "Government authorities confirmed that ",
        "As reported by official wire services, ",
        "In a public communication released this morning, ",
        "A spokesperson for the agency stated that "
    ]
    
    all_suspicious = list(base_suspicious)
    while len(all_suspicious) < target_count:
        msg = random.choice(base_suspicious)
        header = random.choice(suspicious_headers) if random.random() > 0.4 else ""
        footer = random.choice(suspicious_footers) if random.random() > 0.4 else ""
        augmented = f"{header}{msg}{footer}".strip()
        all_suspicious.append(augmented)
        
    all_reliable = list(base_reliable)
    while len(all_reliable) < target_count:
        msg = random.choice(base_reliable)
        prefix = random.choice(reliable_prefixes) if random.random() > 0.5 else ""
        augmented = f"{prefix}{msg[0].lower() + msg[1:] if prefix else msg}".strip()
        all_reliable.append(augmented)
        
    # Build dataframe
    df_suspicious = pd.DataFrame({'text': all_suspicious, 'label': 1, 'category': 'Suspicious'})
    df_reliable = pd.DataFrame({'text': all_reliable, 'label': 0, 'category': 'Reliable'})
    
    df = pd.concat([df_suspicious, df_reliable], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Building curated WhatsApp and News benchmark dataset...")
    df = augment_data(SUSPICIOUS_MESSAGES, RELIABLE_MESSAGES, target_count=350)
    
    raw_path = os.path.join(raw_dir, 'whatsapp_and_news_dataset.csv')
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset: {raw_path} ({len(df)} samples)")
    print("Class distribution:")
    print(df['category'].value_counts())
    
    # Train / Test split (80/20 stratified)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train set: {train_path} ({len(train_df)} samples)")
    print(f"Saved test set: {test_path} ({len(test_df)} samples)")

if __name__ == '__main__':
    main()
