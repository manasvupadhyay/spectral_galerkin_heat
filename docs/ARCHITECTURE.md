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
flowchart TD
    %% Styles
    classDef client fill:#f5f5f5,stroke:#333,stroke-width:2px;
    classDef interface fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,stroke-dasharray: 5 5;
    classDef factory fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef product fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% Client Layer
    subgraph ClientLayer [Client Side]
        Main[main.py]:::client
        Workflow[SimulationWorkflow]:::client
        Context[SimulationContext]:::client
        
        Main --> Context
        Main --> Workflow
    end

    %% Interface Layer
    subgraph InterfaceLayer [Interfaces]
        ISimFactory["<< Interface >>\nSimulationFactory"]:::interface
        IHeat["<< Interface >>\nHeatSolver"]:::interface
        IMicro["<< Interface >>\nMicrostructureSolver"]:::interface
    end

    %% Implementation Layer
    subgraph ImplementationLayer [Concrete Implementations]
        direction LR
        
        %% Factories
        CPUFact[CPUSimulationFactory]:::factory
        GPUFact[GPUSimulationFactory]:::factory
        
        %% Products
        CPUSolv[SpectralCPUSolver]:::product
        GPUSolv[SpectralGPUSolver]:::product
        TreeSolv[TreeMicroSolver]:::product
    end

    %% Relationships
    %% Client uses Interfaces
    Workflow -->|Uses| ISimFactory
    Workflow -->|Calls| IHeat
    Workflow -->|Calls| IMicro

    %% Factories Implement Interface
    CPUFact -.->|Implements| ISimFactory
    GPUFact -.->|Implements| ISimFactory

    %% Factories Create Products
    CPUFact -->|Creates| CPUSolv
    CPUFact -->|Creates| TreeSolv
    GPUFact -->|Creates| GPUSolv

    %% Products Implement Interfaces
    CPUSolv -.->|Implements| IHeat
    GPUSolv -.->|Implements| IHeat
    TreeSolv -.->|Implements| IMicro
```
