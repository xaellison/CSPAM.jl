module CuSparseArgExtrema

    include("int_encoding.jl")
    export monotonic_reinterpret, retreive_arg

    include("reflection.jl")
    export generate_compound_shfl
end
