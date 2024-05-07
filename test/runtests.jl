using CuSparseArgExtrema
using Test
using CUDA


@testset "CuSparseArgExtrema.jl" begin
    
    @testset "int_encoding" begin
        
        function create_test_array()
            # numbers that we definitely want in the test
            special_numbers = [-Inf32, 0.0f0, Inf32]
            # include rands at various orders of magnitude
            A = foldl(vcat, rand(Float32, 1000) * (2.0f0^i) .* 2 .- 1 for i = -100:5:100)
            A = vcat(A, special_numbers)
            A
        end

        function test_reversibility()
            A = create_test_array()
            if !all(monotonic_reinterpret.(Float32, monotonic_reinterpret.(UInt32, A)) .== A)
                failure = findfirst(i -> monotonic_reinterpret(Float32, monotonic_reinterpret(UInt32, A[i])) != A[i], 1:length(A))
                @error "Fails for float = $(A[failure])"
                return false
            end
            true
        end
        @test test_reversibility()

        function test_comparison()
            A = create_test_array()
            B = create_test_array()
            try
                e_A = map(a->monotonic_reinterpret(UInt32, a), A)
                e_B = map(b->monotonic_reinterpret(UInt32, b), B)
                for (x, e_x) in zip(A, e_A)
                    for (y, e_y) in zip(B, e_B)
                        if (x < y) != (e_x < e_y)
                            @error "Failed for $x < $y)"
                            return false
                        end
                    end
                end
            catch E
                # debug note: output may directly reparse as float64
                @error E
                return false
            end

            return true
        end
        @test test_comparison()

        function test_comparison2()
            N = 100000
            A = UInt32.(rand(UInt32, N) .% (1 << 24) .+ rand(UInt32, N) .% (1 << 31))
            B = UInt32.(rand(UInt32, N) .% (1 << 24) .+ rand(UInt32, N) .% (1 << 31))
            try
                e_A = map(a->monotonic_reinterpret(Float32, a), A)
                e_B = map(b->monotonic_reinterpret(Float32, b), B)
                for (x, e_x) in zip(A, e_A)
                    for (y, e_y) in zip(B, e_B)
                        if !isnan(e_x) & !isnan(e_y) & (x < y) != (e_x < e_y)
                            @error "Failed for $x < $y != $e_x < $e_y"
                            return false
                        end
                    end
                end
            catch E
                # debug note: output may directly reparse as float64
                @error E
                return false
            end

            return true
        end
        @test test_comparison2()

    end

    @testset "reflection" begin
        
        struct A
            a::Int
            b::Tuple{Float32, Float32}
        end
        
        generate_compound_shfl(A)

        function k(c::AbstractArray{T}) where T
            c[threadIdx().x] = CuSparseArgExtrema.compound_shfl(c[threadIdx().x])
            return
        end

        a0 = A.(rand(Int, 32), tuple.(rand(Float32, 32), rand(Float32, 32)))
        c0 = cu(a0)
        c = copy(c0)
        CUDA.@sync @cuda threads=32 k(c)
        a = Array(c)
        
        @test all(a[1 + (i) % 32] == a0[1 + (i + 1) % 32] for i in 1:32)
    end
end
