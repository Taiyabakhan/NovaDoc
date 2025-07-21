import os
from local_vector_store import LocalVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()

class QueryEngine:
    def __init__(self):
        # Initialize local vector store
        print("Loading vector store for search...")
        self.vector_store = LocalVectorStore()
        
        # Initialize text generation model (runs locally!)
        print("Loading text generation model...")
        model_name = "facebook/blenderbot-400M-distill"  # Good for Q&A
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Models loaded successfully!")

    def search_documents(self, question, top_k=5):
        """Search for relevant documents using the question"""
        try:
            # Preprocess the question for better search results
            processed_question = self.preprocess_question(question)

            # Lower threshold to be more inclusive
            #search_results = self.vector_store.search(processed_question, k=top_k, score_threshold=0.1)
            # Updated threshold strategy
            search_results = self.vector_store.search(processed_question, k=top_k, score_threshold=0.45)

            # Optional fallback for edge cases
            if len(search_results) == 0:
                print("⚠️ No results found with high threshold. Trying relaxed search...")
                search_results = self.vector_store.search(processed_question, k=top_k, score_threshold=0.3)


            # Format results
            relevant_chunks = []
            for result in search_results:
                relevant_chunks.append({
                    'text': result['text'],
                    'source': result['metadata'].get('source', 'unknown.txt'),  # FIXED
                    'score': result['score']
                })


            return relevant_chunks

        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def generate_answer_simple(self, question, relevant_chunks):
        """Generate a simple answer using template matching (fallback)"""
        if not relevant_chunks:
            return "⚠️ I couldn't find relevant information to answer your question. Please try rephrasing or ensure the document is loaded."

        # Combine relevant context
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        context_lower = context.lower()
        question_lower = question.lower()

        # Helper function to match any keyword
        def match_keywords(keywords):
            return any(keyword in question_lower for keyword in keywords)

        def keywords_in_context(keywords):
            return any(keyword in context_lower for keyword in keywords)

        # Template responses based on category
        if match_keywords(["vacation", "time off", "pto"]):
            if keywords_in_context(["vacation", "time off", "pto"]):
                if "15 days" in context or "20 vacation days" in context:
                    answer = "Based on our company policy, full-time employees are entitled to 15–20 days of paid vacation per year. Vacation must be requested 2 weeks in advance through the HR portal. Up to 5 days may be carried over."
                else:
                    answer = f"📌 Vacation Policy:\n{context[:400]}..."
            else:
                answer = "⚠️ Vacation policy information was not clearly found in the current documents."

        elif match_keywords(["expense", "reimburse", "travel", "cost"]):
            if keywords_in_context(["expense", "reimburse", "travel"]):
                answer = "For expense reimbursement, submit expenses within 30 days with receipts. Covered items include travel, meals, and lodging. Use the finance portal to submit requests."
            else:
                answer = "⚠️ Expense policy details were not located in the context."

        elif match_keywords(["it", "tech", "support", "help", "technical", "computer"]):
            if keywords_in_context(["support", "help", "tech", "it", "portal"]):
                answer = "For IT support, contact help@techcorp.com or call (555) 123-TECH. The emergency IT line (555-911-TECH) is available 24/7. Visit portal.techcorp.com for routine requests."
            else:
                answer = "⚠️ No IT support details found in the matched content."

        elif match_keywords(["remote", "work from home", "telework"]):
            if keywords_in_context(["remote", "work from home", "telework"]):
                answer = "Employees can work remotely up to 3 days per week with manager approval. A stable internet connection is required. Video call attire should remain professional."
            else:
                answer = "⚠️ Remote work information is missing in the current search result."

        elif match_keywords(["holiday", "public holiday", "office closed"]):
            if keywords_in_context(["holiday", "new year's day", "thanksgiving"]):
                answer = f"📆 Company Holidays:\n{context[:400]}..."
            else:
                answer = "⚠️ Couldn't find holiday details in the current documents."

        elif match_keywords(["dress code", "attire", "what to wear"]):
            if keywords_in_context(["dress", "attire", "jeans", "business casual"]):
                answer = f"👔 Dress Code Policy:\n{context[:400]}..."
            else:
                answer = "⚠️ Dress code policy was not clearly found. Please verify the document includes this section."

        elif match_keywords(["sick leave", "sick days", "illness", "health"]):
            if keywords_in_context(["sick", "illness", "doctor"]):
                answer = f"🤒 Sick Leave Policy:\n{context[:400]}..."
            else:
                answer = "⚠️ Sick leave details were not located in the current content."

        elif match_keywords(["performance review", "appraisal", "evaluation", "check-in"]):
            if keywords_in_context(["review", "check-in", "evaluation", "goal setting"]):
                answer = f"📈 Performance Review Schedule:\n{context[:400]}..."
            else:
                answer = "⚠️ No performance review information found in the matched content."

        elif match_keywords(["who", "contact", "call", "email"]):
            if keywords_in_context(["email", "contact", "call", "@", "phone"]):
                answer = f"📞 Contact Info:\n{context[:400]}..."
            else:
                answer = "⚠️ Contact details not found in the current document sections."

        else:
            # Use top chunk for fallback
            top_chunk = relevant_chunks[0]
            source_name = os.path.basename(top_chunk.get('source', 'unknown.txt'))
            answer = f"{top_chunk['text'][:300]}...\n\n📚 Source: {source_name}"


        # Add source information
        sources = list(set(chunk.get('source', 'unknown.txt') for chunk in relevant_chunks))
        if sources:
            source_names = [os.path.basename(src) for src in sources]
            answer += f"\n\n📚 Sources: {', '.join(source_names)}"

        return answer


    def ask_question(self, question, use_advanced=False):
        """Main method to ask a question and get an answer"""
        print(f"Question: {question}")

        # Search with high threshold first
        relevant_chunks = self.vector_store.search(
            self.preprocess_question(question),
            k=5,
            score_threshold=0.45
        )

        # Relax threshold only if no results
        if not relevant_chunks:
            print("⚠️ No high-confidence matches found. Trying relaxed threshold...")
            relevant_chunks = self.vector_store.search(
                self.preprocess_question(question),
                k=5,
                score_threshold=0.3
            )

        print(f"Found {len(relevant_chunks)} relevant chunks")

        if not relevant_chunks:
            return "⚠️ Sorry, I couldn't find a confident answer to your question. Try rephrasing or check if the document is loaded."

        # Warn if low match score
        if relevant_chunks[0]['score'] < 0.45:
            print("⚠️ Warning: Top result has a low confidence score.")

        # Answer generation
        try:
            if use_advanced:
                answer = self.generate_answer_advanced(question, relevant_chunks)
            else:
                answer = self.generate_answer_simple(question, relevant_chunks)
        except Exception as e:
            print(f"⚠️ Error generating answer: {str(e)}. Falling back to simple method.")
            answer = self.generate_answer_simple(question, relevant_chunks)

        return answer



    def generate_answer_advanced(self, question, relevant_chunks):
        """Generate answer using HuggingFace model with RAG prompt"""
        if not relevant_chunks:
            return self.generate_answer_simple(question, relevant_chunks)
        
        try:
            # Combine top 3 chunks for richer context
            top_chunks = relevant_chunks[:3]
            context = "\n\n".join([chunk['text'].strip() for chunk in top_chunks])
            
            # Improved RAG-style prompt
            prompt = f"""
    You are an assistant answering questions based on the provided company documentation.

    Context:
    {context}

    Question: {question}
    Answer:"""

            # Tokenize prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

            # Create Attention Mask
            attention_mask = (inputs != 0).long()  # Create attention mask

            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,  # Pass the attention mask
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            # Decode model output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer after "Answer:" or fallback
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            # Fallback to simple generation if the model fails
            if len(answer) < 10:
                return self.generate_answer_simple(question, relevant_chunks)

            # Append sources
            sources = list(set([chunk['source'] for chunk in relevant_chunks]))
            if sources:
                source_names = [os.path.basename(src) for src in sources]
                answer += f"\n\n📚 Sources: {', '.join(source_names)}"

            return answer

        except Exception as e:
            print(f"Error with advanced generation, falling back to simple: {str(e)}")
            return self.generate_answer_simple(question, relevant_chunks)

    
    def get_vector_store_stats(self):
        """Get vector store statistics"""
        return self.vector_store.get_stats()
            
    # Add to query_engine.py for better question understanding
    def preprocess_question(self, question):
        """Preprocess question for better search results"""
        # Expand common abbreviations
        abbreviations = {
            "PTO": "paid time off vacation",
            "HR": "human resources",
            "IT": "information technology technical support",
            "FAQ": "frequently asked questions",
        }
        
        processed_question = question
        for abbrev, expansion in abbreviations.items():
            processed_question = processed_question.replace(abbrev, expansion)
        
        return processed_question

    def get_question_category(self, question):
        """Categorize question to improve search strategy"""
        question_lower = question.lower()
        
        categories = {
            'hr_policy': ['vacation', 'leave', 'benefits', 'policy', 'hr'],
            'it_support': ['password', 'login', 'computer', 'software', 'it', 'technical'],
            'finance': ['expense', 'reimburse', 'budget', 'cost', 'payment'],
            'general': []
        }
        
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
