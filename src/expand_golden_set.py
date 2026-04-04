import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import get_strategy1_retriever, get_strategy2_retriever

load_dotenv()

QUESTION_TEMPLATES = {
    "financial_figures": [
        "What was Berkshire Hathaway's total revenue in {year}?",
        "What was Berkshire's net earnings in {year}?",
        "What was Berkshire's cash position in {year}?",
        "What was Berkshire's total assets in {year}?",
        "What was GEICO's revenue in {year}?",
        "What was BNSF's revenue in {year}?",
        "What was Berkshire's insurance float in {year}?",
        "What was Berkshire's operating earnings in {year}?",
        "What was Berkshire's book value per share in {year}?",
        "What was Berkshire's investment portfolio value in {year}?",
    ],
    "business_overview": [
        "What are Berkshire Hathaway's main business segments in {year}?",
        "How does Berkshire make money from insurance in {year}?",
        "What is BNSF and what does it do?",
        "What is Berkshire Hathaway Energy?",
        "How does Berkshire approach acquisitions?",
        "What is Berkshire's investment philosophy?",
        "What manufacturing businesses does Berkshire own?",
        "What retail businesses does Berkshire own?",
        "How does Berkshire's insurance float work?",
        "What is Berkshire's competitive advantage?",
    ],
    "risk_factors": [
        "What cybersecurity risks does Berkshire mention in {year}?",
        "What climate risks does Berkshire face in {year}?",
        "What regulatory risks does Berkshire mention in {year}?",
        "What are the main risks to Berkshire's insurance business?",
        "What are the main risks to BNSF railroad?",
        "What geopolitical risks does Berkshire mention?",
        "What interest rate risks does Berkshire face?",
        "What competition risks does Berkshire mention?",
        "What are the risks to Berkshire's energy business?",
        "What are the risks to Berkshire's investment portfolio?",
    ],
    "company_info": [
        "Who is the CEO of Berkshire Hathaway?",
        "Where is Berkshire Hathaway headquartered?",
        "How many employees does Berkshire Hathaway have in {year}?",
        "What stock exchanges is Berkshire listed on?",
        "Who is on Berkshire's board of directors?",
        "What is Berkshire's stock ticker symbol?",
        "When was Berkshire Hathaway founded?",
        "Who is Warren Buffett?",
        "What is Berkshire's corporate structure?",
        "How many subsidiaries does Berkshire have?",
    ],
    "strategy": [
        "What is Berkshire's approach to capital allocation?",
        "How does Berkshire evaluate potential acquisitions?",
        "What is Berkshire's dividend policy?",
        "How does Berkshire approach share buybacks?",
        "What is Berkshire's investment strategy?",
        "How does Berkshire manage its subsidiaries?",
        "What criteria does Berkshire use to hire managers?",
        "How does Berkshire approach debt financing?",
        "What is Berkshire's approach to ESG?",
        "How does Berkshire think about long-term value creation?",
    ],
    "investments": [
        "What is Berkshire's largest stock holding in {year}?",
        "What is Berkshire's investment in Apple in {year}?",
        "What is Berkshire's investment in Bank of America in {year}?",
        "What is Berkshire's investment in Coca Cola in {year}?",
        "What is Berkshire's investment in American Express in {year}?",
        "How much did Berkshire invest in stocks in {year}?",
        "What stocks did Berkshire buy in {year}?",
        "What stocks did Berkshire sell in {year}?",
        "What is the total value of Berkshire's stock portfolio in {year}?",
        "How does Berkshire select its stock investments?",
        "What is Berkshire's largest equity position in {year}?",
        "How much cash does Berkshire hold for investments in {year}?",
        "What is Berkshire's return on investment in {year}?",
        "How does Berkshire think about stock market valuations?",
        "What new investments did Berkshire make in {year}?",
        "Did Berkshire reduce any positions in {year}?",
        "What is Berkshire's dividend income from investments in {year}?",
        "How does Berkshire manage its investment portfolio?",
        "What percentage of Berkshire's portfolio is in financials in {year}?",
        "What is Berkshire's unrealized gain on investments in {year}?",
    ],
    "operations": [
        "How many claims did GEICO process in {year}?",
        "What was BNSF's freight volume in {year}?",
        "How many cars did Berkshire's auto dealerships sell in {year}?",
        "What was Berkshire Energy's power generation capacity in {year}?",
        "How did Berkshire's manufacturing businesses perform in {year}?",
        "What was Berkshire's underwriting profit in {year}?",
        "How much did Berkshire spend on capital expenditures in {year}?",
        "What was Berkshire's insurance loss ratio in {year}?",
        "How many insurance policies does GEICO have in {year}?",
        "What was BNSF's operating ratio in {year}?",
        "How did Berkshire's retail businesses perform in {year}?",
        "What was Berkshire's return on equity in {year}?",
        "How much did Berkshire pay in taxes in {year}?",
        "What was Berkshire's free cash flow in {year}?",
        "How did Berkshire's service businesses perform in {year}?",
        "What was the combined ratio for Berkshire's insurance in {year}?",
        "How much did Berkshire spend on acquisitions in {year}?",
        "What was Berkshire's depreciation expense in {year}?",
        "How many properties does Berkshire own in {year}?",
        "What was Berkshire's inventory level in {year}?",
    ],
    "market_position": [
        "What is GEICO's market share in {year}?",
        "How does BNSF compare to other railroads in {year}?",
        "What is Berkshire's competitive position in insurance in {year}?",
        "How does Berkshire Energy compare to other utilities in {year}?",
        "What is Berkshire's brand value in {year}?",
        "How does Berkshire's size compare to competitors in {year}?",
        "What market segments does Berkshire dominate in {year}?",
        "How does Berkshire's float compare to competitors in {year}?",
        "What is Berkshire's pricing power in insurance in {year}?",
        "How does BNSF's efficiency compare to competitors in {year}?",
        "What barriers to entry does Berkshire have in {year}?",
        "How does Berkshire's investment returns compare to S&P 500 in {year}?",
        "What is Berkshire's market capitalization in {year}?",
        "How does Berkshire's cash generation compare to peers in {year}?",
        "What makes Berkshire unique compared to other conglomerates in {year}?",
        "How does Berkshire's underwriting discipline compare to peers?",
        "What is Berkshire's credit rating in {year}?",
        "How does Berkshire's capital structure compare to peers in {year}?",
        "What strategic advantages does Berkshire have in {year}?",
        "How does Berkshire's management compare to other companies?",
    ],
    "esg_governance": [
        "What is Berkshire's approach to climate change in {year}?",
        "What renewable energy investments does Berkshire have in {year}?",
        "How does Berkshire approach corporate governance in {year}?",
        "What is Berkshire's diversity and inclusion policy in {year}?",
        "How does Berkshire approach environmental sustainability in {year}?",
        "What is Berkshire's executive compensation structure in {year}?",
        "How does Berkshire approach social responsibility in {year}?",
        "What is Berkshire's approach to board oversight in {year}?",
        "How much does Berkshire invest in renewable energy in {year}?",
        "What is Berkshire's carbon footprint in {year}?",
        "How does Berkshire approach data privacy in {year}?",
        "What is Berkshire's approach to employee welfare in {year}?",
        "How does Berkshire handle conflicts of interest in {year}?",
        "What is Berkshire's whistleblower policy in {year}?",
        "How does Berkshire approach anti-corruption in {year}?",
        "What is Berkshire's approach to supply chain ethics in {year}?",
        "How does Berkshire measure ESG performance in {year}?",
        "What sustainability goals has Berkshire set for {year}?",
        "How does Berkshire report on ESG metrics in {year}?",
        "What is Berkshire's approach to stakeholder engagement in {year}?",
    ],
    "historical": [
        "How has Berkshire's revenue grown from 2020 to 2024?",
        "How has Berkshire's float grown over the past 5 years?",
        "What was Berkshire's best performing year between 2020 and 2024?",
        "How has GEICO's performance changed from 2020 to 2024?",
        "How has BNSF's performance changed from 2020 to 2024?",
        "What major acquisitions did Berkshire make between 2020 and 2024?",
        "How has Berkshire's cash position changed from 2020 to 2024?",
        "How has Berkshire's book value grown from 2020 to 2024?",
        "What was Berkshire's compound annual growth rate from 2020 to 2024?",
        "How has Berkshire's employee count changed from 2020 to 2024?",
        "How did COVID-19 impact Berkshire's business in 2020?",
        "How did Berkshire recover from COVID-19 impacts by 2021?",
        "What were Berkshire's biggest challenges between 2020 and 2024?",
        "How has Berkshire's investment portfolio changed from 2020 to 2024?",
        "What were Berkshire's biggest successes between 2020 and 2024?",
        "How has Berkshire's operating earnings trended from 2020 to 2024?",
        "What strategic shifts did Berkshire make between 2020 and 2024?",
        "How has Berkshire's insurance business evolved from 2020 to 2024?",
        "What lessons did Berkshire learn from 2020 to 2024?",
        "How has Berkshire's competitive position changed from 2020 to 2024?",
    ],
}

YEARS = ["2020", "2021", "2022", "2023", "2024"]

def generate_questions():
    questions = []
    qid = 21
    for category, templates in QUESTION_TEMPLATES.items():
        for template in templates:
            if "{year}" in template:
                for year in YEARS:
                    questions.append({
                        "id": qid,
                        "question": template.format(year=year),
                        "category": category,
                        "year": year,
                        "expected": ""
                    })
                    qid += 1
            else:
                questions.append({
                    "id": qid,
                    "question": template,
                    "category": category,
                    "year": "any",
                    "expected": ""
                })
                qid += 1
    return questions

def generate_expected_answers(questions, sample_size=50):
    print(f"Generating expected answers for {sample_size} questions...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Based on this context about Berkshire Hathaway, answer this question very briefly.
    Provide only 2-3 key words or a short phrase that should appear in a correct answer.
    Be specific to the year mentioned in the question if applicable.
    
    Context: {context}
    Question: {question}
    
    Respond with only the key words/phrase, nothing else.
    """)

    sample = questions[:sample_size]
    seen_answers = {}  # Track answers per question pattern to detect duplicates

    for q in sample:
        # Use year-specific retriever if question has a year
        if q.get("year") and q["year"] != "any":
            retriever = get_strategy2_retriever(year_filter=q["year"], k=6)
        else:
            retriever = get_strategy1_retriever(k=6)

        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            expected = chain.invoke(q["question"])
            expected = expected.strip()

            # Detect duplicate answers for same question pattern across years
            base_q = q["question"].replace(q.get("year", ""), "YEAR")
            if base_q in seen_answers:
                if seen_answers[base_q] == expected:
                    print(f"⚠️  Q{q['id']}: Duplicate answer detected — marking as needs_review")
                    q["expected"] = expected
                    q["needs_review"] = True
                else:
                    q["expected"] = expected
                    q["needs_review"] = False
            else:
                q["expected"] = expected
                q["needs_review"] = False

            seen_answers[base_q] = expected
            flag = "⚠️ " if q.get("needs_review") else "✅"
            print(f"{flag} Q{q['id']} [{q.get('year','any')}]: {q['question'][:45]}... -> {expected}")

        except Exception as e:
            q["expected"] = "berkshire"
            q["needs_review"] = True
            print(f"❌ Q{q['id']}: Error - {e}")

    return questions
def run():
    print("=" * 60)
    print("EXPANDING GOLDEN SET TO 500+ QUESTIONS")
    print("=" * 60)
    questions = generate_questions()
    print(f"\nGenerated {len(questions)} questions across {len(QUESTION_TEMPLATES)} categories")
    print("\nBy category:")
    from collections import Counter
    cats = Counter(q["category"] for q in questions)
    for cat, count in cats.items():
        print(f"  {cat}: {count} questions")
    questions = generate_expected_answers(questions, sample_size=50)
    with open("expanded_golden_set.json", "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\n✅ Saved {len(questions)} questions to expanded_golden_set.json")
    print(f"✅ {sum(1 for q in questions if q.get('expected'))} questions have expected answers")
    duplicates = sum(1 for q in questions if q.get('needs_review'))
    print(f"⚠️  {duplicates} questions flagged for review")

if __name__ == "__main__":
    run()