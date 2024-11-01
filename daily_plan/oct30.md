1. Code Development:
Set up genmo and diffusion model
Goals:
Deep understanding of code logic structure
Understanding fine-tuning principles, techniques, and operations
2. Literature Review:
Focused on papers about generating data from videos
Key areas:
Combination of diffusion models and VLM (Vision Language Models)
Flow diffusion concepts (partially reviewed)
Trajectory prediction using ODE (Ordinary Differential Equations)
Good solution for handling occluded trajectories
3. Research Direction Planning:
Theme: "Diffusion Models for Robot Data Generation"
Main Concept:
approach = {
    "tool": "Video diffusion",
    "goal": "Generate robot training data videos",
    "constraint": "Match physical world momentum/dynamics",
    "process": {
        "step1": "Use existing video + robot data",
        "step2": "Generate more robot videos via diffusion",
        "step3": "Create robot-specific diffusion model",
        "starting_point": "Single-arm robots (simpler case)"
    }
}
Tomorrow's Plan:
Machine Learning Fundamentals (2 hours)
Study unfamiliar terms
Review concepts
Tweet interesting/valuable findings
Diffusion Model Deep Dive (4 hours)
Code-level analysis
Related Literature Review (4 hours)
Cross-domain paper reading