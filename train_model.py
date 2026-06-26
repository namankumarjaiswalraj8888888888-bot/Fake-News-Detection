"""
train_model.py - Train and save the Fake News Detection ML model
================================================================
This script:
  1. Generates / downloads a labelled fake-news dataset
  2. Preprocesses the text
  3. Vectorises it with TF-IDF
  4. Trains a Logistic Regression classifier
  5. Evaluates accuracy and prints a classification report
  6. Saves model.pkl and vectorizer.pkl for use in app.py

Run:  python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from utils import clean_text   # our custom text-cleaner

# ─── 1. CREATE / LOAD DATASET ─────────────────────────────────────────────────

def create_dataset() -> pd.DataFrame:
    """
    Build a balanced synthetic dataset that mimics real-world fake-news corpora.
    In a real deployment you would replace this with the WELFake or LIAR dataset
    loaded via pandas:  pd.read_csv("dataset.csv")
    
    Labels:  0 = Real News,  1 = Fake News
    """
    real_news = [
        "Scientists at NASA have confirmed the discovery of water ice on the Moon's south pole, opening possibilities for future lunar missions.",
        "The World Health Organization has approved a new vaccine for malaria, the first in history, after decades of research.",
        "Global temperatures have risen 1.1 degrees Celsius above pre-industrial levels, according to the latest IPCC report.",
        "The United Nations Security Council passed a resolution calling for an immediate ceasefire in the conflict zone.",
        "Apple reported quarterly earnings of $90 billion, beating analyst expectations by 5 percent.",
        "A new study published in Nature found that regular exercise reduces the risk of heart disease by up to 35 percent.",
        "The European Central Bank raised its benchmark interest rate by half a percentage point to combat inflation.",
        "Amazon announced plans to hire 100,000 workers across the United States over the next two years.",
        "Researchers at MIT have developed a new battery technology that could double the range of electric vehicles.",
        "The Federal Aviation Administration approved the first commercial flying taxi service in the United States.",
        "Stock markets rose sharply after the Federal Reserve signaled a pause in interest rate hikes.",
        "The United States and China agreed to resume high-level trade talks after months of diplomatic tension.",
        "A magnitude 6.2 earthquake struck central Italy, causing structural damage but no reported casualties.",
        "The British Parliament approved new legislation aimed at reducing carbon emissions by 78 percent by 2035.",
        "Pfizer announced positive results from a Phase 3 trial of its new cancer immunotherapy drug.",
        "Google reported a 20 percent increase in revenue driven by strong advertising sales.",
        "The International Monetary Fund lowered its global growth forecast to 2.8 percent for the year.",
        "Scientists have identified a new species of dinosaur in Argentina that lived 98 million years ago.",
        "The Supreme Court upheld the constitutionality of the Affordable Care Act in a 7-2 ruling.",
        "NASA's James Webb Space Telescope captured the most detailed images of the early universe to date.",
        "The unemployment rate fell to 3.4 percent, the lowest level in over 50 years.",
        "Tesla delivered a record 422,000 vehicles in the second quarter, exceeding Wall Street expectations.",
        "A new variant of the influenza virus has been detected in Southeast Asia, prompting WHO monitoring.",
        "The G7 nations agreed to impose a price cap of $60 per barrel on Russian crude oil.",
        "Archaeologists discovered an ancient Roman mosaic beneath a construction site in Naples, Italy.",
        "Microsoft announced a $10 billion investment in OpenAI to accelerate artificial intelligence research.",
        "The Paris Agreement's carbon reduction targets are not being met by most signatory nations, a UN report found.",
        "A record 2.3 billion people watched the FIFA World Cup final, according to broadcast figures.",
        "The FDA approved a new drug for Alzheimer's disease that slows cognitive decline by 35 percent.",
        "South Korea's economy contracted by 0.4 percent in the third quarter amid weak exports.",
        "Astronomers detected radio signals from a galaxy 3 billion light years away using the FAST telescope.",
        "India surpassed China to become the world's most populous country, according to UN data.",
        "The Bank of England raised interest rates to a 15-year high to control persistent inflation.",
        "Electric vehicle sales in Europe reached a record 25 percent market share in the first quarter.",
        "The Pentagon confirmed the first successful test of a hypersonic missile capable of Mach 5 speeds.",
        "Australia declared a climate emergency and committed to net-zero emissions by 2035.",
        "A new Oxford University study found that Mediterranean diets reduce dementia risk by 23 percent.",
        "OPEC+ members agreed to cut oil production by 1.66 million barrels per day starting in May.",
        "The Mars Perseverance rover collected its 23rd rock sample, expanding the geological record of Mars.",
        "Japan's government approved a record defence budget of $51 billion, the largest in the nation's history.",
        "Moderna released Phase 2 data showing its mRNA flu vaccine is 86 percent effective.",
        "The World Bank approved a $1.5 billion loan package for flood-ravaged Pakistan.",
        "Ghana became the first country in the world to approve a malaria vaccine for broad use.",
        "The Biden administration announced $369 billion in clean energy investments under the Inflation Reduction Act.",
        "Scientists in South Korea set a new nuclear fusion record, sustaining plasma for 30 seconds.",
        "Twitter agreed to pay a $150 million fine to settle allegations it misused user data for advertising.",
        "The New York Times reported that cybersecurity breaches cost US companies $8.1 billion last year.",
        "Brazil's Amazon deforestation fell by 67 percent in the first year of the Lula administration.",
        "Archaeologists found a 3,000-year-old city beneath the Mediterranean Sea off the coast of Greece.",
        "The UN Food and Agriculture Organization reported that global hunger affects 828 million people.",
    ]

    fake_news = [
        "BREAKING: Government microchips found in COVID vaccines that track your location in real time!",
        "Scientists ADMIT the Earth is actually flat! NASA has been lying to us for decades about space!",
        "CURE FOUND: Drinking bleach mixed with lemon juice kills all viruses instantly, Big Pharma hiding it!",
        "LEAKED: Bill Gates and George Soros planning to reduce world population by 90% using chemtrails!",
        "5G towers causing coronavirus outbreak! Telecom companies installing them at night to infect us!",
        "The moon landing was filmed in a Hollywood studio. NASA whistleblower leaks the truth!",
        "EXPOSED: Tap water contains fluoride that lowers IQ and makes people obedient to the government!",
        "Aliens living among us in government! Top-secret documents reveal extraterrestrial leaders of major nations!",
        "SHOCKING: Doctors earning millions to push dangerous vaccines, natural herbs cure everything!",
        "The deep state is using weather control machines to cause hurricanes and destroy red states!",
        "BREAKING: Democrats installing voting machines programmed to always flip Republican votes to Democrat!",
        "Obama founded ISIS and Hillary Clinton ran a child trafficking ring from a pizza restaurant!",
        "CONFIRMED: George Soros paying protestors $1500 a day to cause riots across America!",
        "Secret society of reptilian shapeshifters controls the global banking system! Wake up sheeple!",
        "COVID-19 was created in a Wuhan lab and released on purpose to destroy the global economy!",
        "Hollywood elites drinking children's blood in secret rituals to stay young and powerful!",
        "LEAKED DOCUMENTS: The sun is actually a giant computer simulation and NASA knows it!",
        "Fluoride in water supply is a mind control drug approved by the United Nations globalists!",
        "EXPOSED: Chemtrails contain dangerous chemicals that cause cancer — government admits secretly!",
        "The New World Order has placed satellites to beam mind-control signals into everyone's brains!",
        "SHOCKING TRUTH: Cancer has been cured but hospitals hide it to keep making billions from chemo!",
        "Climate change is a HOAX invented by China to destroy American manufacturing jobs!",
        "BOMBSHELL: Antifa is funded by billionaires to start a communist revolution in the United States!",
        "Secret underground cities built by elite to survive a planned nuclear war they are starting next year!",
        "Pizzagate is real! Tunnels under Washington D.C. used by politicians to traffic children!",
        "PROOF: Dinosaurs never existed — fossils are planted by museums to teach fake evolution!",
        "The FDA is suppressing a miracle drug that cures diabetes in 72 hours using simple herbs!",
        "BREAKING: George Soros controls all major media outlets and pays journalists to spread propaganda!",
        "Sandy Hook was staged with crisis actors. No children actually died — government false flag!",
        "LEAKED: The Vatican controls the world's governments through a secret council of shadow rulers!",
        "SHOCKING: Scientists paid by Big Oil to fake climate data and spread the global warming hoax!",
        "Elon Musk is actually a clone! The real Elon was replaced by a globalist puppet in 2018!",
        "Mind control frequencies hidden in pop music turn young people into government zombies!",
        "The Great Reset is a plan by the WEF to eliminate 80% of the human population by 2030!",
        "BREAKING: COVID vaccines contain nanobots that activate when 5G towers reach full power!",
        "Chemtrails proven to cause infertility in women — governments depopulating the planet secretly!",
        "EXPOSED: Anthony Fauci is a paid agent of the Chinese Communist Party sabotaging America!",
        "Secret tunnel network under the United States used by Satanic elites for child trafficking!",
        "TRUTH REVEALED: The pyramids were built by aliens using anti-gravity technology 50,000 years ago!",
        "BOMBSHELL: Mainstream media receiving billions from Soros to hide the truth from the public!",
        "The banking cabal prints fake money to control governments and enslave the global population!",
        "EXCLUSIVE: Scientists reveal the Earth's core is hollow and contains a hidden civilisation!",
        "Big Pharma paying doctors to misdiagnose patients so they can sell more unnecessary medication!",
        "LEAKED: The WHO is a criminal organisation planning mandatory global vaccinations with kill switches!",
        "Donald Trump won 2020 by 30 million votes — the steal has been mathematically proven!",
        "SHOCKING: Monsanto created COVID-19 in a secret lab to sell their new patented antidote!",
        "NASA hiding giant alien mothership spotted orbiting Jupiter in new telescope images!",
        "BOMBSHELL: Hospital protocols deliberately killing COVID patients to inflate death statistics!",
        "The HAARP facility in Alaska is used to control the weather and create natural disasters on demand!",
        "EXPOSED: World leaders at Davos agreed to poison food supply to reduce global population by half!",
    ]

    # Combine into a DataFrame
    texts  = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)   # 0 = real, 1 = fake

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    return df


# ─── 2. MAIN TRAINING ROUTINE ─────────────────────────────────────────────────

def train_and_save():
    print("=" * 60)
    print("   Fake News Detection — Model Training")
    print("=" * 60)

    # --- Load / create data ---
    csv_path = "dataset.csv"
    if os.path.exists(csv_path):
        print(f"\n[+] Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("\n[+] Generating built-in dataset …")
        df = create_dataset()
        df.to_csv(csv_path, index=False)
        print(f"    Saved to {csv_path}")

    print(f"    Total samples : {len(df)}")
    print(f"    Real news     : {(df['label'] == 0).sum()}")
    print(f"    Fake news     : {(df['label'] == 1).sum()}")

    # --- Clean text ---
    print("\n[+] Cleaning text …")
    df["clean_text"] = df["text"].apply(clean_text)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.20,    # 20 % held out for testing
        random_state=42,
        stratify=df["label"],   # keep class balance in both splits
    )
    print(f"    Training samples : {len(X_train)}")
    print(f"    Test samples     : {len(X_test)}")

    # --- TF-IDF Vectoriser ---
    # TF-IDF converts raw text into numerical feature vectors.
    # n-gram range (1,2) captures single words AND two-word phrases.
    vectorizer = TfidfVectorizer(
        max_features=10000,    # vocabulary size cap
        ngram_range=(1, 2),    # unigrams and bigrams
        stop_words="english",  # remove common words (the, is, …)
        sublinear_tf=True,     # apply log scaling to term frequencies
    )

    print("\n[+] Fitting TF-IDF vectoriser …")
    X_train_vec = vectorizer.fit_transform(X_train)   # fit on training data ONLY
    X_test_vec  = vectorizer.transform(X_test)        # transform test data

    # --- Logistic Regression Classifier ---
    # Logistic Regression is fast, interpretable, and works well on TF-IDF features.
    model = LogisticRegression(
        C=1.0,            # regularisation strength (lower C = stronger regularisation)
        max_iter=1000,    # max iterations for solver convergence
        solver="lbfgs",   # efficient solver for multiclass / binary problems
        random_state=42,
    )

    print("[+] Training Logistic Regression model …")
    model.fit(X_train_vec, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"   Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save model and vectoriser ---
    print("\n[+] Saving model and vectoriser …")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("    model.pkl      ✓")
    print("    vectorizer.pkl ✓")
    print("\n[✓] Training complete! Ready to run app.py\n")


if __name__ == "__main__":
    train_and_save()
