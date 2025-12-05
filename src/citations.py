"""Citations and References for Publication.

This module contains all necessary citations for datasets, methods, and architectures
used in the DFT→GNN→QNN hybrid approach. All citations follow standard academic format
and include DOI links where available.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class Citation:
    """Academic citation with all required information."""
    authors: str
    title: str
    journal: str
    year: int
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    url: Optional[str] = None
    bibtex_key: Optional[str] = None


class Citations:
    """Collection of all citations used in the work."""
    
    # Dataset Citations
    JARVIS_DFT = Citation(
        authors="Choudhary, K. and DeCost, B.",
        title="Atomistic Line Graph Neural Network for improved materials property predictions",
        journal="npj Computational Materials",
        year=2021,
        volume="7",
        pages="185",
        doi="10.1038/s41524-021-00650-1",
        bibtex_key="choudhary2021jarvis"
    )
    
    JARVIS_ELASTIC = Citation(
        authors="Choudhary, K. and DeCost, B. and Tavazza, F.",
        title="Machine learning with force-field inspired descriptors for materials: fast screening and mapping energy landscape",
        journal="Physical Review Materials",
        year=2018,
        volume="2",
        pages="083801",
        doi="10.1103/PhysRevMaterials.2.083801",
        bibtex_key="choudhary2018jarvis"
    )
    
    OC20_S2EF = Citation(
        authors="Chanussot, L. and Das, A. and Goyal, S. and Lavril, T. and Shuaibi, M. and Riviere, M. and Tran, K. and Heras-Domingo, J. and Ho, C. and Hu, W. and Palizhati, A. and Sriram, A. and Wood, B. and Yoon, J. and Parikh, D. and Zitnick, C. L. and Das, D.",
        title="Open Catalyst 2020 (OC20) Dataset and Community Challenges",
        journal="ACS Catalysis",
        year=2021,
        volume="11",
        pages="6059-6072",
        doi="10.1021/acscatal.0c04525",
        bibtex_key="chanussot2021oc20"
    )
    
    OC22_S2EF = Citation(
        authors="Tran, R. and Lan, J. and Shuaibi, M. and Wood, B. M. and Goyal, S. and Das, A. and Heras-Domingo, J. and Kolluru, A. and Rizvi, A. and Shoghi, N. and Sriram, A. and Therrien, A. and Abed, M. and Vazquez-Mayagoitia, A. and Chen, S. and Acerson, L. and Andonian, R. and Ong, S. P. and Zitnick, C. L. and Ulissi, A. W.",
        title="The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts",
        journal="ACS Catalysis",
        year=2023,
        volume="13",
        pages="3066-3084",
        doi="10.1021/acscatal.2c05426",
        bibtex_key="tran2023oc22"
    )
    
    ANI1X = Citation(
        authors="Smith, J. S. and Isayev, O. and Roitberg, A. E.",
        title="ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost",
        journal="Chemical Science",
        year=2017,
        volume="8",
        pages="3192-3203",
        doi="10.1039/C6SC05720A",
        bibtex_key="smith2017ani1"
    )
    
    # Model Architecture Citations
    SCHNET = Citation(
        authors="Schütt, K. and Kindermans, P. J. and Sauceda Felix, H. E. and Chmiela, S. and Tkatchenko, A. and Müller, K. R.",
        title="SchNet: A continuous-filter convolutional neural network for modeling quantum interactions",
        journal="Advances in Neural Information Processing Systems",
        year=2017,
        volume="30",
        pages="991-1001",
        arxiv="1712.06113",
        bibtex_key="schutt2017schnet"
    )
    
    DOMAIN_AWARE_SCHNET = Citation(
        authors="Gastegger, M. and Behler, J. and Marquetand, P.",
        title="Machine learning molecular dynamics for the simulation of infrared spectra",
        journal="Chemical Science",
        year=2017,
        volume="8",
        pages="6924-6935",
        doi="10.1039/C7SC02267K",
        bibtex_key="gastegger2017domain"
    )
    
    FILM = Citation(
        authors="Perez, E. and Strub, F. and De Vries, H. and Dumoulin, V. and Courville, A.",
        title="FiLM: Visual Reasoning with a General Conditioning Layer",
        journal="Proceedings of the AAAI Conference on Artificial Intelligence",
        year=2018,
        volume="32",
        pages="3942-3951",
        arxiv="1709.07871",
        bibtex_key="perez2018film"
    )
    
    # Quantum Chemistry Citations
    VQE = Citation(
        authors="Peruzzo, A. and McClean, J. and Shadbolt, P. and Yung, M. H. and Zhou, X. Q. and Love, P. J. and Aspuru-Guzik, A. and O'Brien, J. L.",
        title="A variational eigenvalue solver on a photonic quantum processor",
        journal="Nature Communications",
        year=2014,
        volume="5",
        pages="4213",
        doi="10.1038/ncomms5213",
        bibtex_key="peruzzo2014vqe"
    )
    
    UCCSD = Citation(
        authors="Romero, J. and Babbush, R. and McClean, J. R. and Hempel, C. and Love, P. J. and Aspuru-Guzik, A.",
        title="Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz",
        journal="Quantum Science and Technology",
        year=2019,
        volume="4",
        pages="014008",
        doi="10.1088/2058-9565/aad3e4",
        bibtex_key="romero2019uccsd"
    )
    
    DMET = Citation(
        authors="Knizia, G. and Chan, G. K. L.",
        title="Density Matrix Embedding: A Simple Alternative to Dynamical Mean-Field Theory",
        journal="Physical Review Letters",
        year=2012,
        volume="109",
        pages="186404",
        doi="10.1103/PhysRevLett.109.186404",
        bibtex_key="knizia2012dmet"
    )
    
    # Uncertainty Quantification Citations
    ENSEMBLE_UNCERTAINTY = Citation(
        authors="Lakshminarayanan, B. and Pritzel, A. and Blundell, C.",
        title="Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
        journal="Advances in Neural Information Processing Systems",
        year=2017,
        volume="30",
        pages="6402-6413",
        arxiv="1612.01474",
        bibtex_key="lakshminarayanan2017ensemble"
    )
    
    MC_DROPOUT = Citation(
        authors="Gal, Y. and Ghahramani, Z.",
        title="Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
        journal="Proceedings of the 33rd International Conference on Machine Learning",
        year=2016,
        volume="48",
        pages="1050-1059",
        arxiv="1506.02142",
        bibtex_key="gal2016dropout"
    )
    
    # Machine Learning Framework Citations
    PYTORCH = Citation(
        authors="Paszke, A. and Gross, S. and Massa, F. and Lerer, A. and Bradbury, J. and Chanan, G. and Killeen, T. and Lin, Z. and Gimelshein, N. and Antiga, L. and Desmaison, A. and Kopf, A. and Yang, E. and DeVito, Z. and Raison, M. and Tejani, A. and Chilamkurthy, S. and Steiner, B. and Fang, L. and Bai, J. and Chintala, S.",
        title="PyTorch: An Imperative Style, High-Performance Deep Learning Library",
        journal="Advances in Neural Information Processing Systems",
        year=2019,
        volume="32",
        pages="8024-8035",
        arxiv="1912.01703",
        bibtex_key="paszke2019pytorch"
    )
    
    PYTORCH_GEOMETRIC = Citation(
        authors="Fey, M. and Lenssen, J. E.",
        title="Fast Graph Representation Learning with PyTorch Geometric",
        journal="ICLR Workshop on Representation Learning on Graphs and Manifolds",
        year=2019,
        arxiv="1903.02428",
        bibtex_key="fey2019pytorch"
    )
    
    QISKIT = Citation(
        authors="Qiskit contributors",
        title="Qiskit: An Open-source Framework for Quantum Computing",
        journal="Zenodo",
        year=2021,
        doi="10.5281/zenodo.2573505",
        bibtex_key="qiskit2021"
    )
    
    # Method Citations
    GATE_HARD_RANKING = Citation(
        authors="This work",
        title="Gate-Hard Ranking: A Novel Approach for Selecting Challenging Cases in Hybrid DFT-GNN-QNN Systems",
        journal="Manuscript in preparation",
        year=2024,
        bibtex_key="thiswork2024gatehard"
    )
    
    DELTA_LEARNING = Citation(
        authors="This work",
        title="Delta Learning for Bridging GNN and Quantum Chemistry Predictions",
        journal="Manuscript in preparation",
        year=2024,
        bibtex_key="thiswork2024delta"
    )
    
    HYBRID_PIPELINE = Citation(
        authors="This work",
        title="A Hybrid DFT-GNN-QNN Pipeline for Accurate Molecular Property Prediction",
        journal="Manuscript in preparation",
        year=2024,
        bibtex_key="thiswork2024hybrid"
    )


def get_bibtex_entries() -> str:
    """Generate BibTeX entries for all citations."""
    citations = [
        Citations.JARVIS_DFT,
        Citations.JARVIS_ELASTIC,
        Citations.OC20_S2EF,
        Citations.OC22_S2EF,
        Citations.ANI1X,
        Citations.SCHNET,
        Citations.DOMAIN_AWARE_SCHNET,
        Citations.FILM,
        Citations.VQE,
        Citations.UCCSD,
        Citations.DMET,
        Citations.ENSEMBLE_UNCERTAINTY,
        Citations.MC_DROPOUT,
        Citations.PYTORCH,
        Citations.PYTORCH_GEOMETRIC,
        Citations.QISKIT,
        Citations.GATE_HARD_RANKING,
        Citations.DELTA_LEARNING,
        Citations.HYBRID_PIPELINE
    ]
    
    bibtex_entries = []
    
    for citation in citations:
        entry = f"@{citation.bibtex_key},\n"
        entry += f"  author = {{{citation.authors}}},\n"
        entry += f"  title = {{{citation.title}}},\n"
        entry += f"  journal = {{{citation.journal}}},\n"
        entry += f"  year = {{{citation.year}}},\n"
        
        if citation.volume:
            entry += f"  volume = {{{citation.volume}}},\n"
        if citation.pages:
            entry += f"  pages = {{{citation.pages}}},\n"
        if citation.doi:
            entry += f"  doi = {{{citation.doi}}},\n"
        if citation.arxiv:
            entry += f"  arxiv = {{{citation.arxiv}}},\n"
        if citation.url:
            entry += f"  url = {{{citation.url}}},\n"
        
        entry += "}\n"
        bibtex_entries.append(entry)
    
    return "\n".join(bibtex_entries)


def get_citation_summary() -> Dict[str, List[str]]:
    """Get organized citation summary by category."""
    return {
        "Datasets": [
            "JARVIS-DFT: Choudhary & DeCost (2021)",
            "JARVIS-Elastic: Choudhary et al. (2018)",
            "OC20-S2EF: Chanussot et al. (2021)",
            "OC22-S2EF: Tran et al. (2023)",
            "ANI1x: Smith et al. (2017)"
        ],
        "Model Architectures": [
            "SchNet: Schütt et al. (2017)",
            "Domain-Aware SchNet: Gastegger et al. (2017)",
            "FiLM: Perez et al. (2018)"
        ],
        "Quantum Chemistry": [
            "VQE: Peruzzo et al. (2014)",
            "UCCSD: Romero et al. (2019)",
            "DMET: Knizia & Chan (2012)"
        ],
        "Uncertainty Quantification": [
            "Ensemble Uncertainty: Lakshminarayanan et al. (2017)",
            "MC Dropout: Gal & Ghahramani (2016)"
        ],
        "Software Frameworks": [
            "PyTorch: Paszke et al. (2019)",
            "PyTorch Geometric: Fey & Lenssen (2019)",
            "Qiskit: Qiskit contributors (2021)"
        ],
        "Novel Contributions": [
            "Gate-Hard Ranking: This work (2024)",
            "Delta Learning: This work (2024)",
            "Hybrid Pipeline: This work (2024)"
        ]
    }


def save_citations_to_file(output_path: str = "citations.bib") -> None:
    """Save all citations to a BibTeX file."""
    bibtex_content = get_bibtex_entries()
    
    with open(output_path, 'w') as f:
        f.write("% Citations for DFT-GNN-QNN Hybrid Pipeline\n")
        f.write("% Generated automatically from citations.py\n\n")
        f.write(bibtex_content)
    
    print(f"Citations saved to {output_path}")


def print_citation_summary() -> None:
    """Print organized citation summary."""
    summary = get_citation_summary()
    
    print("=" * 80)
    print("CITATION SUMMARY")
    print("=" * 80)
    
    for category, citations in summary.items():
        print(f"\n{category}:")
        print("-" * 40)
        for citation in citations:
            print(f"  • {citation}")
    
    print("\n" + "=" * 80)
    print("Total citations: 19")
    print("Datasets: 5")
    print("Methods: 8")
    print("Software: 3")
    print("Novel contributions: 3")
    print("=" * 80)


if __name__ == "__main__":
    print_citation_summary()
    save_citations_to_file()


