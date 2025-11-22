"""
Clinical RAG (Retrieval-Augmented Generation) System
Evidence-based clinical knowledge retrieval for AI agents
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from rank_bm25 import BM25Okapi

from healthcare_config import HealthcareConfig
from healthcare_security import audit_logger

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClinicalDocument:
    """Clinical knowledge document"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
        if "last_updated" not in self.metadata:
            self.metadata["last_updated"] = datetime.now().isoformat()
        if "clinical_domain" not in self.metadata:
            self.metadata["clinical_domain"] = "general"


@dataclass
class RetrievalResult:
    """Retrieved clinical evidence"""
    document: ClinicalDocument
    score: float
    rank: int
    retrieval_method: str  # "dense", "sparse", "hybrid"


@dataclass
class ClinicalContext:
    """Augmented clinical context for LLM"""
    query: str
    retrieved_docs: List[RetrievalResult]
    context_text: str
    sources: List[str]
    confidence: float


# ═══════════════════════════════════════════════════════════════════════════
# Clinical Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalKnowledgeBase:
    """
    Clinical knowledge base with common medical knowledge

    In production, this would be populated from:
    - UpToDate
    - Clinical Practice Guidelines
    - Drug databases (RxNorm, DrugBank)
    - LOINC reference ranges
    - ICD-10/SNOMED mappings
    """

    CLINICAL_GUIDELINES = [
        {
            "id": "guideline_001",
            "content": """
            Hypertension Management Guidelines (ACC/AHA 2017):

            Blood pressure categories:
            - Normal: <120/80 mmHg
            - Elevated: 120-129/<80 mmHg
            - Stage 1 HTN: 130-139/80-89 mmHg
            - Stage 2 HTN: ≥140/90 mmHg
            - Hypertensive crisis: >180/120 mmHg

            First-line medications:
            - ACE inhibitors (lisinopril, enalapril)
            - ARBs (losartan, valsartan)
            - Calcium channel blockers (amlodipine)
            - Thiazide diuretics (hydrochlorothiazide)

            Target BP: <130/80 for most patients
            Lifestyle modifications recommended for all patients
            """,
            "metadata": {
                "source": "ACC/AHA Guidelines",
                "clinical_domain": "cardiology",
                "condition": "hypertension",
                "last_updated": "2017-11-13"
            }
        },
        {
            "id": "guideline_002",
            "content": """
            Diabetes Management (ADA 2024):

            HbA1c targets:
            - General target: <7.0%
            - Elderly/comorbidities: <8.0%
            - Healthy adults: <6.5%

            Glucose monitoring:
            - Fasting glucose: 80-130 mg/dL
            - Postprandial: <180 mg/dL

            First-line therapy: Metformin unless contraindicated

            Add-on therapies based on:
            - Cardiovascular disease: GLP-1 RA or SGLT2i
            - Heart failure: SGLT2i
            - CKD: SGLT2i
            - Weight loss goal: GLP-1 RA
            """,
            "metadata": {
                "source": "ADA Standards of Care",
                "clinical_domain": "endocrinology",
                "condition": "diabetes",
                "last_updated": "2024-01-01"
            }
        },
        {
            "id": "vitals_001",
            "content": """
            Normal Vital Sign Ranges:

            Blood Pressure:
            - Systolic: 90-120 mmHg (critical: <70 or >180)
            - Diastolic: 60-80 mmHg (critical: <40 or >120)

            Heart Rate:
            - Adults: 60-100 bpm (critical: <40 or >140)
            - Tachycardia: >100 bpm
            - Bradycardia: <60 bpm

            Respiratory Rate:
            - Adults: 12-20 breaths/min (critical: <8 or >30)

            Temperature:
            - Normal: 36.1-37.2°C / 97-99°F
            - Fever: >38.0°C / 100.4°F
            - Hypothermia: <35.0°C / 95°F

            Oxygen Saturation:
            - Normal: 95-100% (critical: <88%)
            """,
            "metadata": {
                "source": "Clinical Reference",
                "clinical_domain": "general",
                "category": "vital_signs"
            }
        },
        {
            "id": "drug_001",
            "content": """
            Lisinopril (ACE Inhibitor):

            Indications:
            - Hypertension
            - Heart failure
            - Post-MI cardioprotection
            - Diabetic nephropathy

            Dosing:
            - Initial: 10 mg daily
            - Maintenance: 10-40 mg daily
            - Max: 80 mg daily

            Contraindications:
            - Pregnancy (Category D)
            - Angioedema history
            - Bilateral renal artery stenosis

            Adverse effects:
            - Dry cough (10-15%)
            - Hyperkalemia
            - Hypotension
            - Angioedema (rare)

            Monitoring:
            - Serum creatinine, potassium at baseline and 1-2 weeks after initiation
            - Blood pressure
            """,
            "metadata": {
                "source": "Drug Database",
                "clinical_domain": "pharmacology",
                "drug_class": "ACE inhibitor",
                "drug_name": "lisinopril"
            }
        },
        {
            "id": "lab_001",
            "content": """
            Common Lab Test Reference Ranges:

            HbA1c (LOINC: 4548-4):
            - Normal: <5.7%
            - Prediabetes: 5.7-6.4%
            - Diabetes: ≥6.5%

            Potassium (LOINC: 2823-3):
            - Normal: 3.5-5.0 mEq/L
            - Hypokalemia: <3.5 (critical: <2.5)
            - Hyperkalemia: >5.0 (critical: >6.0)

            Creatinine (LOINC: 2160-0):
            - Men: 0.7-1.3 mg/dL
            - Women: 0.6-1.1 mg/dL
            - eGFR calculation uses Cr, age, sex, race

            eGFR stages:
            - Stage 1: >90 (normal)
            - Stage 2: 60-89 (mild CKD)
            - Stage 3a: 45-59 (moderate CKD)
            - Stage 3b: 30-44 (moderate CKD)
            - Stage 4: 15-29 (severe CKD)
            - Stage 5: <15 (kidney failure)
            """,
            "metadata": {
                "source": "Lab Reference",
                "clinical_domain": "laboratory",
                "category": "reference_ranges"
            }
        },
        {
            "id": "risk_001",
            "content": """
            CHADS2-VASc Score for Stroke Risk in Atrial Fibrillation:

            Points:
            - Congestive heart failure: 1
            - Hypertension: 1
            - Age ≥75: 2
            - Diabetes: 1
            - Stroke/TIA history: 2
            - Vascular disease: 1
            - Age 65-74: 1
            - Sex (female): 1

            Interpretation:
            - Score 0 (men) / 1 (women): Low risk, no anticoagulation
            - Score 1 (men) / 2 (women): Moderate risk, consider anticoagulation
            - Score ≥2 (men) / ≥3 (women): High risk, anticoagulation recommended

            Anticoagulation options:
            - DOACs: apixaban, rivaroxaban, edoxaban, dabigatran
            - Warfarin (if DOAC contraindicated)
            """,
            "metadata": {
                "source": "Clinical Algorithm",
                "clinical_domain": "cardiology",
                "algorithm": "CHADS2-VASc",
                "condition": "atrial_fibrillation"
            }
        },
        {
            "id": "high_risk_meds_001",
            "content": """
            High-Risk Medications Requiring Close Monitoring:

            Anticoagulants:
            - Warfarin: INR monitoring
            - DOACs: Renal function, bleeding risk
            - Interactions: NSAIDs, antibiotics

            Insulin and sulfonylureas:
            - Hypoglycemia risk
            - Dose adjustments in renal/hepatic disease

            Opioids:
            - Respiratory depression
            - Constipation management
            - Addiction potential
            - Avoid benzodiazepine combination

            Immunosuppressants:
            - Infection risk
            - Regular lab monitoring
            - Drug level monitoring (tacrolimus, cyclosporine)

            Chemotherapy:
            - Neutropenia monitoring
            - Nausea/vomiting management
            - Specialist supervision required
            """,
            "metadata": {
                "source": "Medication Safety",
                "clinical_domain": "pharmacology",
                "category": "high_risk_medications"
            }
        },
        {
            "id": "polypharmacy_001",
            "content": """
            Polypharmacy Management:

            Definition: Use of ≥5 medications concurrently

            Risks:
            - Drug-drug interactions
            - Adverse drug events
            - Medication non-adherence
            - Falls (especially in elderly)
            - Cognitive impairment

            Medication review checklist:
            1. Is each medication still indicated?
            2. Are there duplicate therapies?
            3. Any drug-drug interactions?
            4. Appropriate for patient's age and renal/hepatic function?
            5. Patient able to afford and adhere?

            Deprescribing candidates:
            - PPIs (if used >8 weeks without indication)
            - Benzodiazepines (in elderly)
            - Anticholinergics (in elderly)
            - Duplicate antihypertensives
            - Supplements without proven benefit
            """,
            "metadata": {
                "source": "Geriatric Medicine",
                "clinical_domain": "pharmacology",
                "category": "polypharmacy"
            }
        }
    ]

    @classmethod
    def get_all_documents(cls) -> List[ClinicalDocument]:
        """Return all clinical knowledge documents"""
        return [
            ClinicalDocument(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
            for doc in cls.CLINICAL_GUIDELINES
        ]


# ═══════════════════════════════════════════════════════════════════════════
# RAG System
# ═══════════════════════════════════════════════════════════════════════════

class ClinicalRAGSystem:
    """
    Retrieval-Augmented Generation system for clinical knowledge

    Uses hybrid retrieval: Dense (embeddings) + Sparse (BM25)
    """

    def __init__(
        self,
        collection_name: str = "clinical_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[Path] = None
    ):
        """
        Initialize RAG system

        Args:
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist vector database
        """
        self.collection_name = collection_name

        # Set up persistent directory
        if persist_directory is None:
            persist_directory = HealthcareConfig.HEALTHCARE_DIR / "vector_db"
        persist_directory.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_directory),
            anonymized_telemetry=False
        ))

        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Clinical knowledge base for RAG"}
            )
            logger.info(f"Created new collection: {collection_name}")

            # Initialize with clinical knowledge
            self._initialize_knowledge_base()

        # BM25 index for sparse retrieval
        self.bm25_index = None
        self.bm25_documents = []
        self._build_bm25_index()

        logger.info("Clinical RAG system initialized")

    def _initialize_knowledge_base(self):
        """Initialize vector database with clinical knowledge"""
        logger.info("Initializing clinical knowledge base...")

        documents = ClinicalKnowledgeBase.get_all_documents()

        self.add_documents(documents)

        logger.info(f"Added {len(documents)} clinical documents to knowledge base")

    def _build_bm25_index(self):
        """Build BM25 index for sparse retrieval"""
        # Get all documents from collection
        results = self.collection.get()

        if results["documents"]:
            self.bm25_documents = results["documents"]

            # Tokenize documents
            tokenized_docs = [doc.lower().split() for doc in self.bm25_documents]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)

            logger.info(f"Built BM25 index with {len(self.bm25_documents)} documents")

    def add_documents(self, documents: List[ClinicalDocument]) -> None:
        """
        Add documents to knowledge base

        Args:
            documents: List of clinical documents to add
        """
        if not documents:
            return

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        # Rebuild BM25 index
        self._build_bm25_index()

        # Audit log
        audit_logger.log_security_event(
            event_type="RAG_KNOWLEDGE_UPDATE",
            severity="INFO",
            description=f"Added {len(documents)} documents to clinical knowledge base"
        )

        logger.info(f"Added {len(documents)} documents to RAG system")

    def retrieve_dense(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Dense retrieval using embeddings

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of retrieval results
        """
        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_metadata
        )

        # Parse results
        retrieval_results = []

        for i in range(len(results["ids"][0])):
            doc = ClinicalDocument(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}
            )

            result = RetrievalResult(
                document=doc,
                score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                rank=i + 1,
                retrieval_method="dense"
            )

            retrieval_results.append(result)

        return retrieval_results

    def retrieve_sparse(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Sparse retrieval using BM25

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of retrieval results
        """
        if not self.bm25_index:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]

        # Create retrieval results
        retrieval_results = []

        for i, idx in enumerate(top_k_indices):
            # Get document from collection
            doc_id = self.collection.get()["ids"][idx]
            doc_content = self.bm25_documents[idx]
            doc_metadata = self.collection.get()["metadatas"][idx] if self.collection.get()["metadatas"] else {}

            doc = ClinicalDocument(
                id=doc_id,
                content=doc_content,
                metadata=doc_metadata
            )

            result = RetrievalResult(
                document=doc,
                score=float(scores[idx]),
                rank=i + 1,
                retrieval_method="sparse"
            )

            retrieval_results.append(result)

        return retrieval_results

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining dense and sparse methods

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for dense retrieval (1-alpha for sparse)
            filter_metadata: Optional metadata filters

        Returns:
            List of retrieval results
        """
        # Get dense results
        dense_results = self.retrieve_dense(query, k=k*2, filter_metadata=filter_metadata)

        # Get sparse results
        sparse_results = self.retrieve_sparse(query, k=k*2)

        # Combine and re-rank
        combined_scores = {}

        # Add dense scores
        for result in dense_results:
            combined_scores[result.document.id] = {
                "score": alpha * result.score,
                "document": result.document
            }

        # Add sparse scores
        for result in sparse_results:
            if result.document.id in combined_scores:
                combined_scores[result.document.id]["score"] += (1 - alpha) * result.score
            else:
                combined_scores[result.document.id] = {
                    "score": (1 - alpha) * result.score,
                    "document": result.document
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:k]

        # Create final results
        retrieval_results = []
        for i, (doc_id, data) in enumerate(sorted_results):
            result = RetrievalResult(
                document=data["document"],
                score=data["score"],
                rank=i + 1,
                retrieval_method="hybrid"
            )
            retrieval_results.append(result)

        return retrieval_results

    def get_clinical_context(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.3,
        filter_domain: Optional[str] = None
    ) -> ClinicalContext:
        """
        Get augmented clinical context for LLM

        Args:
            query: Clinical query
            k: Number of documents to retrieve
            min_score: Minimum relevance score
            filter_domain: Filter by clinical domain

        Returns:
            Clinical context with retrieved evidence
        """
        # Build metadata filter
        filter_metadata = None
        if filter_domain:
            filter_metadata = {"clinical_domain": filter_domain}

        # Retrieve relevant documents
        results = self.retrieve_hybrid(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )

        # Filter by minimum score
        results = [r for r in results if r.score >= min_score]

        # Build context text
        context_parts = []
        sources = []

        for result in results:
            context_parts.append(f"""
            --- Clinical Evidence (Relevance: {result.score:.2f}) ---
            Source: {result.document.metadata.get('source', 'Unknown')}
            Domain: {result.document.metadata.get('clinical_domain', 'general')}

            {result.document.content.strip()}
            """)

            sources.append(result.document.metadata.get('source', 'Unknown'))

        context_text = "\n\n".join(context_parts)

        # Calculate confidence based on relevance scores
        confidence = np.mean([r.score for r in results]) if results else 0.0

        # Audit log
        audit_logger.log_security_event(
            event_type="RAG_RETRIEVAL",
            severity="INFO",
            description=f"Retrieved {len(results)} clinical documents for query"
        )

        return ClinicalContext(
            query=query,
            retrieved_docs=results,
            context_text=context_text,
            sources=list(set(sources)),
            confidence=float(confidence)
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def format_rag_prompt(
    clinical_question: str,
    context: ClinicalContext,
    base_prompt: str
) -> str:
    """
    Format prompt with RAG context

    Args:
        clinical_question: The clinical question
        context: Retrieved clinical context
        base_prompt: Base agent prompt

    Returns:
        Augmented prompt with clinical evidence
    """
    augmented_prompt = f"""
{base_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 CLINICAL EVIDENCE (Retrieved from Knowledge Base)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context.context_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{', '.join(context.sources)}

Evidence Confidence: {context.confidence:.1%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on the above clinical evidence, please answer the following question:

{clinical_question}

Important: Base your response on evidence-based guidelines. If the retrieved
evidence doesn't fully address the question, acknowledge this and provide
general clinical reasoning with appropriate caveats.
"""

    return augmented_prompt


# ═══════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════

# Global RAG instance
_rag_system: Optional[ClinicalRAGSystem] = None


def get_rag_system() -> ClinicalRAGSystem:
    """Get or initialize global RAG system"""
    global _rag_system

    if _rag_system is None:
        _rag_system = ClinicalRAGSystem()

    return _rag_system


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("Clinical RAG System Demo")
    print("=" * 80)

    # Initialize RAG
    rag = ClinicalRAGSystem()

    # Test queries
    test_queries = [
        "What is the normal blood pressure range?",
        "How should I manage a patient with hypertension and diabetes?",
        "What are the high-risk medications that require close monitoring?",
        "What are the dosing guidelines for lisinopril?"
    ]

    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Query: {query}")
        print(f"{'─' * 80}")

        context = rag.get_clinical_context(query, k=2)

        print(f"\nRetrieved {len(context.retrieved_docs)} documents")
        print(f"Confidence: {context.confidence:.1%}")
        print(f"Sources: {', '.join(context.sources)}")

        print("\nTop result:")
        if context.retrieved_docs:
            top_result = context.retrieved_docs[0]
            print(f"  Score: {top_result.score:.3f}")
            print(f"  Content: {top_result.document.content[:200]}...")

    print(f"\n{'=' * 80}")
    print("✅ Clinical RAG System operational")
    print("=" * 80)
