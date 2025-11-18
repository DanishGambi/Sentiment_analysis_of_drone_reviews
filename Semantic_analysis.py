import requests
import json
import re
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
from datetime import datetime


class OllamaDroneReviewAnalyzer:
    def __init__(self, model_names: List[str]):
        self.models = model_names
        self.base_url = "http://localhost:11434/api/generate"
        self.technical_terms = [
            'батарея', 'аккумулятор', 'время полета', 'камера', 'стабилизация',
            'гироскоп', 'gps', 'навигация', 'пульт управления', 'дальность',
            'качество съемки', 'разрешение', '4k', '1080p', 'объектив',
            'светосила', 'автофокус', 'трансмиссия', 'пропеллер', 'мотор',
            'двигатель', 'подвеска', 'сенсор', 'датчик', 'полет',
            'управляемость', 'скорость', 'маневренность', 'зарядка',
            'контроллер', 'приложение', 'софт', 'прошивка', 'калибровка'
        ]

    def load_reviews_from_json(self, file_path: str) -> List[str]:
        """Load reviews from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'reviews' in data:
                return data['reviews']
            else:
                print("Unexpected JSON structure. Returning empty list.")
                return []

        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {file_path}.")
            return []

    def create_prompt(self, review: str) -> str:
        """Create a structured prompt for drone review analysis in Russian"""
        prompt = f"""
        Ты - эксперт по анализу отзывов о дронах. Проанализируй следующий отзыв и предоставь ответ ТОЛЬКО в формате JSON без каких-либо дополнительных объяснений.

        Отзыв: "{review}"

        Проанализируй и определи:
        1. sentiment: общий тон отзыва ("положительный", "отрицательный" или "нейтральный")
        2. main_topic: основная тема обсуждения (например: "камера", "батарея", "время полета", "управление", "стабилизация", "gps", "качество сборки")
        3. issue: конкретная проблема или достоинство (например: "короткое время работы батареи", "отличное качество съемки", "проблемы с подключением")
        4. rating: числовая оценка от 1 до 5, где 1 - очень плохо, 5 - отлично

        Ответ должен быть в точном формате JSON:
        {{
            "sentiment": "положительный",
            "main_topic": "качество камеры", 
            "issue": "короткое время работы батареи",
            "rating": 4
        }}
        """
        return prompt.strip()

    def analyze_with_model(self, model_name: str, review: str, max_retries: int = 3) -> Dict:
        """Analyze review using Ollama model with retry logic"""
        prompt = self.create_prompt(review)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, json=payload, timeout=120)
                response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "").strip()

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    analysis_result = json.loads(json_str)

                    # Validate required fields
                    required_fields = ["sentiment", "main_topic", "issue", "rating"]
                    if all(field in analysis_result for field in required_fields):
                        return analysis_result

                # If we reach here, JSON extraction failed
                print(f"Warning: Failed to extract JSON from {model_name} response. Response: {response_text}")

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {model_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {model_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        # Fallback response
        return {
            "sentiment": "нейтральный",
            "main_topic": "не определено",
            "issue": "не определено",
            "rating": 3
        }

    def analyze_review(self, review: str) -> Dict[str, Dict]:
        """Analyze a single review with all models"""
        results = {}

        for model_name in self.models:
            try:
                print(f"Analyzing with {model_name}...")
                result = self.analyze_with_model(model_name, review)
                results[model_name] = result
                print(f"✓ {model_name} completed")

            except Exception as e:
                print(f"✗ Error analyzing with {model_name}: {e}")
                results[model_name] = {
                    "sentiment": "ошибка",
                    "main_topic": "ошибка",
                    "issue": "ошибка",
                    "rating": 0
                }

        return results

    def batch_analyze_reviews(self, reviews: List[str]) -> List[Dict[str, Dict]]:
        """Analyze multiple reviews"""
        all_results = []

        for i, review in enumerate(reviews):
            print(f"\nAnalyzing review {i + 1}/{len(reviews)}")
            print(f"Review: {review[:100]}...")

            results = self.analyze_review(review)
            all_results.append(results)

            # Small delay to avoid overwhelming the system
            time.sleep(1)

        return all_results


class ModelEvaluator:
    def __init__(self, technical_terms: List[str]):
        self.technical_terms = technical_terms

    def calculate_sentiment_agreement(self, results: List[Dict[str, Dict]]) -> float:
        """Calculate percentage agreement on sentiment between models"""
        agreements = []

        for review_result in results:
            sentiments = [model_result["sentiment"] for model_result in review_result.values()]
            # Check if all models agree
            if len(set(sentiments)) == 1:
                agreements.append(1)
            else:
                agreements.append(0)

        return np.mean(agreements) if agreements else 0.0

    def calculate_technical_term_understanding(self, results: List[Dict[str, Dict]]) -> Dict[str, float]:
        """Evaluate models' ability to understand technical terms"""
        model_scores = {}

        for model_name in next(iter(results)).keys():
            technical_counts = []

            for review_result in results:
                model_output = review_result[model_name]
                # Combine main_topic and issue for analysis
                combined_text = f"{model_output['main_topic']} {model_output['issue']}".lower()

                # Count technical terms found
                found_terms = [term for term in self.technical_terms if term in combined_text]
                technical_counts.append(len(found_terms))

            model_scores[model_name] = np.mean(technical_counts) if technical_counts else 0.0

        return model_scores

    def calculate_rating_consistency(self, results: List[Dict[str, Dict]]) -> Dict[str, float]:
        """Calculate rating consistency (variance) for each model"""
        model_ratings = {}

        for model_name in next(iter(results)).keys():
            ratings = []
            for review_result in results:
                rating = review_result[model_name]["rating"]
                if isinstance(rating, int) and 1 <= rating <= 5:
                    ratings.append(rating)
                elif isinstance(rating, str) and rating.isdigit():
                    ratings.append(int(rating))

            if ratings:
                # Lower variance = more consistent
                variance = np.var(ratings)
                model_ratings[model_name] = variance
            else:
                model_ratings[model_name] = float('inf')

        return model_ratings

    def calculate_response_quality(self, results: List[Dict[str, Dict]]) -> Dict[str, float]:
        """Calculate overall response quality score"""
        quality_scores = {}

        for model_name in next(iter(results)).keys():
            scores = []

            for review_result in results:
                model_output = review_result[model_name]
                score = 0

                # Check for errors
                if "ошибка" in str(model_output.values()):
                    scores.append(0)
                    continue

                # Score sentiment (1 point if valid)
                if model_output["sentiment"] in ["положительный", "отрицательный", "нейтральный"]:
                    score += 1

                # Score main_topic (1 point if not default)
                if model_output["main_topic"] != "не определено":
                    score += 1

                # Score issue (1 point if not default)
                if model_output["issue"] != "не определено":
                    score += 1

                # Score rating (1 point if valid)
                rating = model_output["rating"]
                if (isinstance(rating, int) and 1 <= rating <= 5) or \
                        (isinstance(rating, str) and rating.isdigit() and 1 <= int(rating) <= 5):
                    score += 1

                scores.append(score / 4)  # Normalize to 0-1

            quality_scores[model_name] = np.mean(scores) if scores else 0.0

        return quality_scores


def main():
    # Use your Ollama models
    model_names = [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:3b",
        "qwen2.5-coder:1.5b"
    ]

    # Initialize analyzer
    analyzer = OllamaDroneReviewAnalyzer(model_names)

    # Load reviews from JSON file
    print("Loading reviews from reviews.json...")
    reviews = analyzer.load_reviews_from_json("reviews.json")

    if not reviews:
        print("No reviews found. Using sample reviews instead.")
        # Sample drone reviews in Russian (fallback)
        reviews = [
            "It's a great drone d j I does a wonderful job with their flight computers and gps and motors, and it's flight time versus agility and maneuverability, and the range of expertise that can have a lot of fun and get really good video shots with this drone.It's great for beginners. if you have no experience, try doing a flight simulation on a computer, with a similar controller, or a cheap drone to get the conset and controls down.",
    "I purchased a DJI FPV bundle for $799.00 from this seller. The DJI FPV bundle arrived defective — it powered on and the motors would activate, but as soon as I tried to take off, it displayed an error code and would not fly. I returned the defective DJI FPV bundle exactly as received. The seller refunded only $0.01 and kept $798.99 as a “restocking fee.” Charging a massive restocking fee for a defective DJI FPV bundle is not only unethical but may violate consumer protection laws, including the FTC Act (15 U.S.C. §§ 41–58) and Amazon’s own policy prohibiting excessive return fees for defective products.",
    "Great condition. Came quickly and delivered when I was told it would be. Like new item that is practically brand new. Excellent drone for a variety of uses. I'm enjoying it immensely",
    "I bought this drone for almost $1,000 and it looks like it was used. The glasses don't turn on, they didn't have the protective plastic on the lenses. The box was in very bad condition.",
    "The drone is good but not as easy to get up and running. But straight out of a alien movie! Nice! Lost star since it didnt include the extra battery and an extra pair of knobs for the controller.",
    "Drone came preused, the drone bombs off has some scratches. Had a lot of problems with g. P. S the drone didn't want to fly. It's something I don't know what to say. It was my son birthday and there it goes.",
    "After a short time I had problems with the battery, it stopped receiving a charge.",
    "We bought this drone with the intention of sharing it between dad and son, but so far, it's been my son taking the skies. I've been busy with training modules on the laptop, with mixed success. The drone's speed and agility are truly impressive. We've chased birds, cars, and even wild animals from high above, capturing some amazing footage. Watching the video replay is just as thrilling as flying the drone itself. My son loves the FPV experience, and he can switch between safety mode for beginners and full-speed mode for some seriously wild flying. We’ve sent it so high and far that we lost sight of it, but the return-to-home feature is fantastic—just push a button, and the drone flies back and lands gently. The drone is lightweight, though heavier than expected, and the battery life is robust but short, typically lasting 10-15 minutes. Given the short flight times, extra batteries are a must. You'll also need a special case and potentially a second drone for spare parts, which is par for the course with any hobby. Overall, it's an amazing drone with a few extra costs and considerations. Our next mission is to take it to the middle of nowhere and chase buffalo. Stay tuned for more!",
    "I loved the fact that it was fast I also like the fact that it is compatible with the motion controller. Although I do not like the fact that both times I have purchased the same drone and both times it acted sporadically and during flight actually it was while it was hovering all of a sudden it took off to the left and then back to the right and did that two or three more times in a matter of a couple seconds and then it shot straight up in the air the goggles quit recording at 226 ft but the Drone kept going in the lower right hand corner of the goggles it said battery overload battery overload battery overload flashing and then it went completely dead and came straight down to the ground and landed on my neighbor's roof and destroyed the Drone so after I sent that drone back which UPS seem to lose in the mail and still have not refunded me I bought the Avatar 2 and that drone flew amazingly it was very simple to hook to the motion controller where the DJI fpv was not as easy it was almost like the Avada 2 was built for the motion controller where the DJI fpv made it an absolute hassle and a pain in the butt to hook the motion controller to the DJ ifp anyway both times I ordered the DJI fpv it acted total sporadically like a misbehaved puppy",
    "If you’ve never flown the DJI FPV, watch a tutorial video. Or power on the remote first then the goggles and then the drone last. That’s just how I do it out of habit. Follow the prompts through the goggles, make sure you have the remote the goggle and the drone somewhat close together so you’re sure everything syncs correctly, then once you hear the final beeps, make sure you’re in the comfortable mode, I’d start with (Normal Mode) should be the button on your controller front left if you’re holding it, because it’s the ONLY mode that has the automatic collision avoidance from buildings the ground etc, the other 2 modes (SPORT & MANUAL) DO NOT have that. Stay away from trees, bodies of water & everything else especially if you’re a beginner until you’re comfortable. Then at the same time, take your left and right thumb sticks on your controller down and in towards each other like an L & and backwards L with your right thumb. The motors will start at idle, make sure your props are on correctly!!! I almost lost my drone bc of that reason (my own fault in a hurry) I know the activation part can be very irritating to some. It did frustrate me a little bit but once I got it, boy, oh boy was it worth it! This damn thing is wicked! It’s pretty fast! Try sport mode, or manual mode but just be ready because it will climb altitude so fast and it will go in whatever direction you want to at about 90 miles an hour, it seems like. It is very easy to fly the FPV goggles are my favorite part about it! You may want to sit down or lean up against something while you’re flying because it will throw your equilibrium off a little bit. The quality is top-notch! The performance is top-notch! The range is top-notch! The quality of the connectivity in general is great! It’s got awesome features, like. Return to home automatically, it’ll record a video if it loses signal somehow or you ignore the warnings of low battery and it dies out you can see the last thing it saw right before you lost transmission. Not one bad thing to say about this bad boy. I could fly this thing for hours on end! It’s worth every penny in my opinion. Purchase yourself a couple extra flight batteries when you can it’s worth it! I did. I also bought an extra battery for my goggles as well. I haven’t had to change it out yet (during a 3 flight battery run) - my goggles will be right around 40% still. Same with the remote, it holds power the best out of them all, it’ll usually be around or above 75% after all of this",
    "I'm sorry that I actually bought this ~$800 paperweight. With the FCC drone 'regulations,' I've only been able to fly this thing ONCE. Of the one time I did, it was so incredibly loud even several hundred feet in the air (but of course, can't go above 400 ft, God forbid, regulations and all). I constantly get the 'remote ID not found' error despite paying for the stupid test to get a certificate license to be able to fly IN MY BACKYARD. Not in any restricted zones - my !@#$%$$ backyard. And even though I did everything as instructed, I cannot get it off the ground because of this. This is not the case with every vendor - I'm only aware of DJI kowtowing to this stupidity. Never again, DJI. Wish I never wasted my money on this.",
    "Интересная штука. Быстро дохнет заряд да и сам сдох быстро. Неуклюжая вешь, запуталась у ребёнка в волосах, напугала малышку.",
    "Приехал, и то с трудностями. Отваливается связь с джойстиком, постоянно требуется перезапуск, не летает из-за этого совсем! Крайне не доволен покупкой! Не советую к приобретению!",
    "Долгая доставка. Вводят в заблуждение. На картинке одно по факту другое. На картинке с камерой пришло без камеры. Обман.",
    "Дешёвый, поскольку в нём особо ничего нет, то перепутать что-то сложно. Инструкция на китайском, пришлось как-то нейронками пыхтеть и интернет перемалывать чтобы хоть что-то узнать... Заряжается видимо только от ПК, может быть где-то узнаю какой блок питания к нему нужен... Непонятно можно ли купить какой-то шнур и подключить пульт к симулятору на ПК, ведь без него толку от дрона мало, летает аля 4 минуты, трясётся, летит куда хочет, возможно при доставке погнулись то ли защитные кольца, то ли лопасти то ли ещё что-то, или я так криво доставал, кренит сильно вперёд и вправо. Стики силой пытаются вернуться в исходное положение, так что ни о каком точном полёте речи нет, банально пальцы быстро устают, ещё и пульт махонький"
 ]

    print(f"Loaded {len(reviews)} reviews for analysis.")

    # Analyze reviews
    print("Starting review analysis with Ollama models...")
    results = analyzer.batch_analyze_reviews(reviews)

    # Print results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    for i, (review, review_results) in enumerate(zip(reviews, results)):
        print(f"\n--- Review {i + 1} ---")
        print(f"Review: {review}")
        for model_name, analysis in review_results.items():
            print(f"{model_name}:")
            print(f"  Sentiment: {analysis['sentiment']}")
            print(f"  Main topic: {analysis['main_topic']}")
            print(f"  Issue: {analysis['issue']}")
            print(f"  Rating: {analysis['rating']}")

    # Evaluate models
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    evaluator = ModelEvaluator(analyzer.technical_terms)

    # Sentiment agreement
    sentiment_agreement = evaluator.calculate_sentiment_agreement(results)
    print(f"\nSentiment Agreement between models: {sentiment_agreement:.2%}")

    # Technical term understanding
    technical_scores = evaluator.calculate_technical_term_understanding(results)
    print("\nTechnical Term Understanding (avg terms per review):")
    for model, score in technical_scores.items():
        print(f"  {model}: {score:.2f}")

    # Rating consistency
    rating_consistency = evaluator.calculate_rating_consistency(results)
    print("\nRating Consistency (variance, lower is better):")
    for model, variance in rating_consistency.items():
        print(f"  {model}: {variance:.2f}")

    # Response quality
    quality_scores = evaluator.calculate_response_quality(results)
    print("\nOverall Response Quality (0-1 scale):")
    for model, score in quality_scores.items():
        print(f"  {model}: {score:.2f}")

    # Create comparison table
    df = create_comparison_table(results, reviews)
    print(f"\nComparison table shape: {df.shape}")

    # Save results to JSON
    output_data = {
        "models_used": model_names,
        "reviews": reviews,
        "analysis_results": results,
        "evaluation_metrics": {
            "sentiment_agreement": sentiment_agreement,
            "technical_understanding": technical_scores,
            "rating_consistency": rating_consistency,
            "response_quality": quality_scores
        }
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"drone_reviews_ollama_analysis_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Save comparison table to CSV
    df.to_csv(f"drone_reviews_comparison_{timestamp}.csv", index=False, encoding='utf-8')

    print(f"\nResults saved to 'drone_reviews_ollama_analysis.json'")
    print(f"Comparison table saved to 'drone_reviews_comparison.csv'")
    print(f"Total reviews analyzed: {len(reviews)}")


def create_comparison_table(results: List[Dict[str, Dict]], reviews: List[str]):
    """Create a pandas DataFrame for easy comparison"""
    comparison_data = []

    for i, (review, review_results) in enumerate(zip(reviews, results)):
        row = {"review_id": i + 1, "review_text": review}
        for model_name, analysis in review_results.items():
            for key, value in analysis.items():
                row[f"{model_name}_{key}"] = value
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df


if __name__ == "__main__":
    main()