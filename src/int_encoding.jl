# Linearly encode float32s into uint32 in a way that preserves comparison & equality
# for all normal non-NaN values


function monotonic_reinterpret(::Type{UInt32}, f::Float32)::UInt32
    # equivalent with float comparison
    #f < 0 ? ~reinterpret(UInt32, f) : reinterpret(UInt32, f) + UInt32(1<<31)
    uint = reinterpret(UInt32, f)
    uint > 1 << 31 ? ~uint : uint | UInt32(1 << 31)
end

function monotonic_reinterpret(::Type{UInt64}, f_i :: Tuple{Float32, UInt32})::UInt64
    f, i = f_i
    return UInt64(monotonic_reinterpret(UInt32, f)) << 32 | UInt64(i)
end

function retreive_arg(i64::UInt64)::UInt32
    # for an UInt64 encoding a (max/min, argmax/argmin), return argmax/argmin
    return UInt32(i64 ⊻ (i64 >> 32 << 32))
end

function monotonic_reinterpret(::Type{Float32}, i32::UInt32)::Float32
    flipped_sign = i32 >> 31 << 31
    if flipped_sign == 0
        # originally, sign 0 positive, so this is negative, restore
        return reinterpret(Float32, ~i32)
    else
        return reinterpret(Float32, i32 ⊻ UInt32(1 << 31))
    end 
end
