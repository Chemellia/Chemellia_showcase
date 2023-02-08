### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0180b07f-0319-4bfd-9f20-415f021aa049
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
	
    using PlutoUI, AtomsBase, AtomsIO, AtomGraphs, ChemistryFeaturization, DataFrames
	using ChemistryFeaturization.ElementFeature
end

# ╔═╡ 95450e24-a1a3-11ed-2b36-c9ef4bfe3dd0
md"""
# Chemellia Showcase!
This notebook steps through some aspects of the Chemellia ecosystem for the case of graph-based representations and models. It also highlights integrations with other Julia ecosystems and interfaces that allow Chemellia itself to be quite lightweight.
"""

# ╔═╡ 76cb16d4-61eb-4db4-8574-247078043ba1
TableOfContents()

# ╔═╡ a55e9ca6-0396-4dc1-afd8-379dc03063fe
md"""
## Building Graphs
The first step is to go from a full 3D representation of a crystal structure to the graph-based one that we'll use as input to our model. We'll start out playing with HoPt$_3$, a structure I pulled from the [Materials Project](https://materialsproject.org/) database. A unit cell looks like this:

$(PlutoUI.LocalResource("./mp-195.png"))

Let's read in the CIF file as an `AtomGraph` and take a look around...
"""

# ╔═╡ a41c48b6-3a4c-4bf5-9cfd-81388424a20d
HoPt₃ = AtomGraph("mp-195.cif")

# ╔═╡ 5b68df70-f055-4069-a35b-c173870e984e
propertynames(HoPt₃)

# ╔═╡ 71e8634e-16d9-4422-81bf-8e33f29fd487
md"It stores both the `graph` that we built, and the representation of the full 3D `structure` that was generated along the way. That structure makes use of the [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) interface for easy interoperability of representations. The graph is a `SimpleWeightedGraph` from the [JuliaGraphs](https://juliagraphs.org/) ecosystem, and makes use of the Graphs.jl interface. Let's visualize it:"

# ╔═╡ 10d1afae-6bf4-4914-bfaa-43f02e956c4b
visualize(HoPt₃)

# ╔═╡ f7a6ae01-2e6d-4c16-aaf8-9af17a0e59e8
md"""
(The `visualize` function in AtomGraphs.jl makes use of the GraphPlot.jl package, another package within the JuliaGraphs ecosystem.)

The AtomGraphs.jl package is capable of reading from a variety of file formats for representing atoms and molecules, and can also directly take any object that is an AtomsBase `AbstractSystem` (such as a structure used in a [DFTK](https://docs.dftk.org/stable/) or [Molly](https://github.com/JuliaMolSim/Molly.jl) calculation).

We can also do some customization of how the graphs are built. To explore that, let's look at another crystal structure, this time with a somewhat larger unit cell...

$(PlutoUI.LocalResource("./VInO4.png"))

Let's play a bit with what the cutoff radius does to the connectivity of the graph and the "apparent" coordination numbers of the atoms/nodes...
"""

# ╔═╡ 578e1a80-9499-42db-8ed3-da1e0713e604
cutoff_slider = @bind cutoff Scrubbable(2.75:0.25:4, format=".2f")

# ╔═╡ c20fb391-1765-4b29-9eb8-7f2c8f4351bf
begin
	VInO₄_smallcutoff = AtomGraph("./VInO4.cif", cutoff_radius=cutoff)
	visualize(VInO₄_smallcutoff)
end

# ╔═╡ 7e3ad6e5-323b-4e29-ba58-6c243a416ea9
md"""
#### Other graph-building options
* You can tweak the behavior of the cutoff method to generate edges via some more keyword arguments. `max_num_nbr` is reasonably self-explanatory. There is also `dist_decay_func`, which sets how the edge weights decay with separation distance.
* Another way to build the graph, rather than the somewhat naïve cutoff method, is to construct Voronoi tessellations of the structure and build edges only between atoms that share a face. This option is turned on by passing the keyword argument `use_voronoi=true`
"""

# ╔═╡ 660b7b62-8edf-4bdc-9d6c-c1b46002590c
md"""
## Featurization
Once we've built our graph representation the way we like it, we can featurize it, "annotating" it with information that we would want to feed into an ML model. Let's step through some of the "vocabulary" around featurization within Chemellia. In particular, it's handled within the interface provided by the ChemistryFeaturization package.

### Feature Descriptors
The main building block of a featurization is a **feature descriptor** object, which should be a subtype of `AbstractFeatureDescriptor`. For now, we'll focus `ElementFeatureDescriptor`s, which only need elemental information in order to compute their values.

To start, suppose we're interested in featurizing our graphs with information about which block of the periodic table (s, p, d, f) an element is in. We can construct the feature descriptor like so:
"""

# ╔═╡ a55da127-e349-42fe-9280-e1c868c025f7
block = ElementFeatureDescriptor("Block")

# ╔═╡ c731b3fc-8b39-40a6-9b75-5e021795a495
md"""
We can "call" the feature descriptor directly on an `AtomGraph` to get its value for each atom... 
"""

# ╔═╡ 6f69dffb-57c2-410e-9e4a-9c2eb62124cf
block(HoPt₃)

# ╔═╡ 0e077478-0bf6-45ca-8969-22c4ad009b08
md"What other element features are built in?"

# ╔═╡ 9d589e76-c65c-4fc1-b013-265237677cb9
elementfeature_names

# ╔═╡ 03426b8e-4535-4e3d-bb81-73e067214100


# ╔═╡ 0656f790-f295-48a5-8a18-f671dd1db925
begin
	sixp = ElementFeatureDescriptor("6p")
	encodable_elements(sixp)
end

# ╔═╡ e18cdb44-4e20-4e93-a065-638bde8318b0
md"""
Those are the only elements for which this feature is defined! Nifty.

We can also easily define custom features. For an `ElementFeatureDescriptor`, all that's needed is a lookup table defining  Check this out:
"""

# ╔═╡ b73dc3d3-ee38-46c7-a268-e47c0d692c9a
begin
	lookup_table = DataFrame(["Ho" 73; "Pt" 92; "In" 126; "V" 40], [:Symbol, :Awesomeness])
	awesomeness = ElementFeatureDescriptor("Awesomeness", lookup_table)
end

# ╔═╡ 87aedd6e-9af1-442d-93bd-6b1a45838315
awesomeness(HoPt₃)

# ╔═╡ bef802d7-9715-4c11-bb73-de2949cbc0fc
awesomeness(VInO₄_smallcutoff)

# ╔═╡ fc4625bb-9b16-4fc8-8b8e-175758115aeb
md"""
...and there you see the usefulness of the `encodable_elements` functionality.

### Codecs

"""

# ╔═╡ 175f5541-1929-4d2f-aca0-82be15890738


# ╔═╡ Cell order:
# ╟─95450e24-a1a3-11ed-2b36-c9ef4bfe3dd0
# ╟─76cb16d4-61eb-4db4-8574-247078043ba1
# ╠═0180b07f-0319-4bfd-9f20-415f021aa049
# ╟─a55e9ca6-0396-4dc1-afd8-379dc03063fe
# ╠═a41c48b6-3a4c-4bf5-9cfd-81388424a20d
# ╠═5b68df70-f055-4069-a35b-c173870e984e
# ╟─71e8634e-16d9-4422-81bf-8e33f29fd487
# ╠═10d1afae-6bf4-4914-bfaa-43f02e956c4b
# ╟─f7a6ae01-2e6d-4c16-aaf8-9af17a0e59e8
# ╟─578e1a80-9499-42db-8ed3-da1e0713e604
# ╟─c20fb391-1765-4b29-9eb8-7f2c8f4351bf
# ╟─7e3ad6e5-323b-4e29-ba58-6c243a416ea9
# ╟─660b7b62-8edf-4bdc-9d6c-c1b46002590c
# ╠═a55da127-e349-42fe-9280-e1c868c025f7
# ╟─c731b3fc-8b39-40a6-9b75-5e021795a495
# ╠═6f69dffb-57c2-410e-9e4a-9c2eb62124cf
# ╟─0e077478-0bf6-45ca-8969-22c4ad009b08
# ╠═9d589e76-c65c-4fc1-b013-265237677cb9
# ╠═03426b8e-4535-4e3d-bb81-73e067214100
# ╠═0656f790-f295-48a5-8a18-f671dd1db925
# ╠═e18cdb44-4e20-4e93-a065-638bde8318b0
# ╠═b73dc3d3-ee38-46c7-a268-e47c0d692c9a
# ╠═87aedd6e-9af1-442d-93bd-6b1a45838315
# ╠═bef802d7-9715-4c11-bb73-de2949cbc0fc
# ╠═fc4625bb-9b16-4fc8-8b8e-175758115aeb
# ╠═175f5541-1929-4d2f-aca0-82be15890738
