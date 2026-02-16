# Architecture & Design

The project is structured around the **Abstract Factory** pattern, enabling easy extension to new backends (CPU, GPU, distributed) and numerical methods (Spectral, FEM, etc.).

## Directory Structure

```text
fastHeatSolv/
├── main.py                     # Entry point: parses args, instantiates SimulationFactory
├── config/                     # YAML configuration files for simulation parameters
├── data/
│   └── paths/                  # G-code files for laser paths
├── out/                        # Simulation results (per-run subfolders)
├── core/
│   ├── workflow.py             # Simulation loop (orchestrator, strategy pattern)
│   ├── parameters.py           # Data classes: PhysParams, NumParams, GeomParams, IOParams
│   └── io.py                   # Abstract base class for IOManager
├── interfaces/
│   ├── factory.py              # Abstract Factory: interface for creating solvers, IO managers
│   └── solver.py               # Abstract Product: HeatSolver interface
├── implementations/
│   ├── factories/              # Concrete Factories (CPU, GPU, FEM)
│   ├── solvers/                # Concrete Solvers (Spectral, FEM wrappers)
│   ├── file_io/                # Concrete IO Managers
│   └── physics/                # Low-level physics kernels
└── utils/                      # Utilities (DCT, visualization, logging)
```

## Component Descriptions

- **Client (`main.py`)**: Entry point that parses arguments (CLI/YAML) and injects the appropriate Factory into the Workflow.
- **Workflow (`core/workflow.py`)**: The high-level director that manages the time loop, physics updates, and IO events.
- **Factory Interface (`interfaces/factory.py`)**: Abstract definitions for creating solvers and managers.
- **Solvers (`implementations/solvers/`)**: The mathematical engines.

## Extending the Framework

To add a new solver (e.g., Finite Difference):
1. Create `interfaces/solver.py` compliant implementation in `implementations/solvers/`.
2. Create a factory in `implementations/factories/`.
3. Register the method in `main.py`.

## Architecture Diagram

```mermaid
graph TD
    %% Main Entry Points
    Main[main.py]:::main
    Context[SimulationContext<br/>Params]:::note
    
    Main --> Context

    %% Core Logic Container
    subgraph Core[Core Logic]
        style Core fill:#f5f5f5,stroke:#666666,stroke-width:2px
        Workflow[SimulationWorkflow]:::core
        RunLoop((run loop)):::core
        Workflow --> RunLoop
    end
    
    %% Main Connections
    Context -.-> Workflow
    Main --> Workflow

    %% Abstract Factory Pattern
    subgraph FactoryPattern[Abstract Factory Pattern]
        style FactoryPattern fill:#ffffff,stroke:#666666,stroke-width:2px,stroke-dasharray: 5 5

        %% Interfaces
        ISimFactory["<< Interface >><br/>SimulationFactory"]:::interface
        IHeatSolver["<< Interface >><br/>HeatSolver"]:::interface
        IMicroSolver["<< Interface >><br/>MicrostructureSolver"]:::interface
        IIOManager["<< Interface >><br/>IOManager"]:::interface

        %% Concrete Factories
        CPUFact[CPUSimulationFactory]:::impl
        GPUFact[GPUSimulationFactory]:::impl

        %% Concrete Solvers
        CPUSolver[SpectralCPUSolver]:::impl
        GPUSolver[SpectralGPUSolver]:::impl
        MicroSolver[TreeMicroSolver]:::impl

        %% Relationships
        CPUFact -.->|implements| ISimFactory
        GPUFact -.->|implements| ISimFactory
        
        CPUFact -.->|creates| CPUSolver
        GPUFact -.->|creates| GPUSolver
        CPUFact -.->|creates| MicroSolver
        
        CPUSolver -.->|implements| IHeatSolver
        GPUSolver -.->|implements| IHeatSolver
        MicroSolver -.->|implements| IMicroSolver
    end

    %% Workflow usage of factories
    Workflow --> ISimFactory
    RunLoop --> IHeatSolver
    RunLoop --> IMicroSolver

    %% Solving Scheme Details
    subgraph Scheme[Solving Scheme Details]
        style Scheme fill:#e1f5fe,stroke:#01579b

        Init[1. Initialize<br/>Load Neper/MicrostructPy]:::step
        TimeLoop{2. Time Loop<br/>while t < t_end}:::decision
        HeatStep[3a. Heat Solver<br/>Compute T Field]:::step
        MicroStep[3b. Microstructure<br/>Update]:::micro
        IOCheck[4. IO Check]:::step
        Log[5. Logging / ETA]:::step
        
        %% Microstructure Detail logic
        MicroLogic["<b>Microstructure Logic:</b><br/>1. Get Isotherm (T_melt)<br/>2. Query AABB Tree (Melt Zone)<br/>3. Remove melted seeds<br/>4. Project seeds to front<br/>5. Evolve & Write new seeds"]:::note

        Init --> TimeLoop
        TimeLoop --> HeatStep
        HeatStep --> MicroStep
        MicroStep --> IOCheck
        IOCheck --> Log
        Log --> TimeLoop
        
        MicroStep -.-> MicroLogic
    end

    %% Classes
    classDef main fill:#f5f5f5,stroke:#666666,font-weight:bold;
    classDef note fill:#fff2cc,stroke:#d6b656,stroke-width:1px;
    classDef core fill:#dae8fc,stroke:#6c8ebf,font-weight:bold;
    classDef interface fill:#d5e8d4,stroke:#82b366,font-style:italic;
    classDef impl fill:#ffffff,stroke:#82b366,stroke-width:2px;
    classDef step fill:#e1d5e7,stroke:#9673a6;
    classDef decision fill:#f8cecc,stroke:#b85450,shape:rhombus;
    classDef micro fill:#ffe6cc,stroke:#d79b00,font-weight:bold;
```
