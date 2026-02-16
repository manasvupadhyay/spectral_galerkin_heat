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
    %% --- Theme & Styling ---
    classDef container fill:#2d2d2d,stroke:#555,color:#fff;
    classDef client fill:#ff7675,stroke:#d63031,stroke-width:2px,color:#000;
    classDef interface fill:#81ecec,stroke:#00cec9,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef factory fill:#55efc4,stroke:#00b894,stroke-width:2px,color:#000;
    classDef product fill:#fdcb6e,stroke:#e17055,stroke-width:2px,color:#000;

    %% --- Main Architecture Container ---
    subgraph Architecture [System Architecture]
        style Architecture fill:#333333,stroke:#666,color:#fff
        
        %% Client Layer
        subgraph ClientLayer [Client Side]
            style ClientLayer fill:#404040,stroke:#777,color:#fff
            Main[main.py]:::client
            Workflow[SimulationWorkflow]:::client
            Context[SimulationContext]:::client
            
            Main --> Context
            Main --> Workflow
        end

        %% Interface Layer
        subgraph InterfaceLayer [Abstract Interfaces]
            style InterfaceLayer fill:#404040,stroke:#777,color:#fff
            ISimFactory["<< Interface >>\nSimulationFactory"]:::interface
            IHeat["<< Interface >>\nHeatSolver"]:::interface
            IMicro["<< Interface >>\nMicrostructureSolver"]:::interface
        end

        %% Implementation Layer
        subgraph ImplementationLayer [Concrete Implementations]
            style ImplementationLayer fill:#404040,stroke:#777,color:#fff
            
            %% Factories
            CPUFact[CPUSimulationFactory]:::factory
            GPUFact[GPUSimulationFactory]:::factory
            
            %% Products
            CPUSolv[SpectralCPUSolver]:::product
            GPUSolv[SpectralGPUSolver]:::product
            TreeSolv[TreeMicroSolver]:::product
        end

        %% Relationships
        Workflow -->|Uses| ISimFactory
        Workflow -->|Calls| IHeat
        Workflow -->|Calls| IMicro

        %% Factory Implementations
        CPUFact -.->|Implements| ISimFactory
        GPUFact -.->|Implements| ISimFactory

        %% Factory Creation Links
        CPUFact -->|Creates| CPUSolv
        CPUFact -->|Creates| TreeSolv
        GPUFact -->|Creates| GPUSolv

        %% Product Implementations
        CPUSolv -.->|Implements| IHeat
        GPUSolv -.->|Implements| IHeat
        TreeSolv -.->|Implements| IMicro
    end
```
