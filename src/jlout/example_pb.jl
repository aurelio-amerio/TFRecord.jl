# Autogenerated using ProtoBuf.jl v1.0.9 on 2022-10-23T13:13:45.095
# original file: /home/zaldivar/Documents/Aurelio/Github/TFRecord.jl/src/proto/example.proto (proto3 syntax)

module example_pb

import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export BytesList, FloatList, Int64List, Feature, Features, FeatureList, Example
export FeatureLists, SequenceExample

struct BytesList
    value::Vector{Vector{UInt8}}
end
PB.default_values(::Type{BytesList}) = (;value = Vector{Vector{UInt8}}())
PB.field_numbers(::Type{BytesList}) = (;value = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:BytesList})
    value = PB.BufferedVector{Vector{UInt8}}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, value)
        else
            PB.skip(d, wire_type)
        end
    end
    return BytesList(value[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::BytesList)
    initpos = position(e.io)
    !isempty(x.value) && PB.encode(e, 1, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::BytesList)
    encoded_size = 0
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 1))
    return encoded_size
end

struct FloatList
    value::Vector{Float32}
end
PB.default_values(::Type{FloatList}) = (;value = Vector{Float32}())
PB.field_numbers(::Type{FloatList}) = (;value = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FloatList})
    value = PB.BufferedVector{Float32}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, value)
        else
            PB.skip(d, wire_type)
        end
    end
    return FloatList(value[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FloatList)
    initpos = position(e.io)
    !isempty(x.value) && PB.encode(e, 1, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FloatList)
    encoded_size = 0
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 1))
    return encoded_size
end

struct Int64List
    value::Vector{Int64}
end
PB.default_values(::Type{Int64List}) = (;value = Vector{Int64}())
PB.field_numbers(::Type{Int64List}) = (;value = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Int64List})
    value = PB.BufferedVector{Int64}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, value)
        else
            PB.skip(d, wire_type)
        end
    end
    return Int64List(value[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Int64List)
    initpos = position(e.io)
    !isempty(x.value) && PB.encode(e, 1, x.value)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Int64List)
    encoded_size = 0
    !isempty(x.value) && (encoded_size += PB._encoded_size(x.value, 1))
    return encoded_size
end

struct Feature
    kind::Union{Nothing,OneOf{<:Union{BytesList,FloatList,Int64List}}}
end
PB.oneof_field_types(::Type{Feature}) = (;
    kind = (;bytes_list=BytesList, float_list=FloatList, int64_list=Int64List),
)
PB.default_values(::Type{Feature}) = (;bytes_list = nothing, float_list = nothing, int64_list = nothing)
PB.field_numbers(::Type{Feature}) = (;bytes_list = 1, float_list = 2, int64_list = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Feature})
    kind = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            kind = OneOf(:bytes_list, PB.decode(d, Ref{BytesList}))
        elseif field_number == 2
            kind = OneOf(:float_list, PB.decode(d, Ref{FloatList}))
        elseif field_number == 3
            kind = OneOf(:int64_list, PB.decode(d, Ref{Int64List}))
        else
            PB.skip(d, wire_type)
        end
    end
    return Feature(kind)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Feature)
    initpos = position(e.io)
    if isnothing(x.kind);
    elseif x.kind.name === :bytes_list
        PB.encode(e, 1, x.kind[]::BytesList)
    elseif x.kind.name === :float_list
        PB.encode(e, 2, x.kind[]::FloatList)
    elseif x.kind.name === :int64_list
        PB.encode(e, 3, x.kind[]::Int64List)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::Feature)
    encoded_size = 0
    if isnothing(x.kind);
    elseif x.kind.name === :bytes_list
        encoded_size += PB._encoded_size(x.kind[]::BytesList, 1)
    elseif x.kind.name === :float_list
        encoded_size += PB._encoded_size(x.kind[]::FloatList, 2)
    elseif x.kind.name === :int64_list
        encoded_size += PB._encoded_size(x.kind[]::Int64List, 3)
    end
    return encoded_size
end

struct Features
    feature::Dict{String,Feature}
end
PB.default_values(::Type{Features}) = (;feature = Dict{String,Feature}())
PB.field_numbers(::Type{Features}) = (;feature = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Features})
    feature = Dict{String,Feature}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, feature)
        else
            PB.skip(d, wire_type)
        end
    end
    return Features(feature)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Features)
    initpos = position(e.io)
    !isempty(x.feature) && PB.encode(e, 1, x.feature)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Features)
    encoded_size = 0
    !isempty(x.feature) && (encoded_size += PB._encoded_size(x.feature, 1))
    return encoded_size
end

struct FeatureList
    feature::Vector{Feature}
end
PB.default_values(::Type{FeatureList}) = (;feature = Vector{Feature}())
PB.field_numbers(::Type{FeatureList}) = (;feature = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FeatureList})
    feature = PB.BufferedVector{Feature}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, feature)
        else
            PB.skip(d, wire_type)
        end
    end
    return FeatureList(feature[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FeatureList)
    initpos = position(e.io)
    !isempty(x.feature) && PB.encode(e, 1, x.feature)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FeatureList)
    encoded_size = 0
    !isempty(x.feature) && (encoded_size += PB._encoded_size(x.feature, 1))
    return encoded_size
end

struct Example
    features::Union{Nothing,Features}
end
PB.default_values(::Type{Example}) = (;features = nothing)
PB.field_numbers(::Type{Example}) = (;features = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:Example})
    features = Ref{Union{Nothing,Features}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, features)
        else
            PB.skip(d, wire_type)
        end
    end
    return Example(features[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::Example)
    initpos = position(e.io)
    !isnothing(x.features) && PB.encode(e, 1, x.features)
    return position(e.io) - initpos
end
function PB._encoded_size(x::Example)
    encoded_size = 0
    !isnothing(x.features) && (encoded_size += PB._encoded_size(x.features, 1))
    return encoded_size
end

struct FeatureLists
    feature_list::Dict{String,FeatureList}
end
PB.default_values(::Type{FeatureLists}) = (;feature_list = Dict{String,FeatureList}())
PB.field_numbers(::Type{FeatureLists}) = (;feature_list = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:FeatureLists})
    feature_list = Dict{String,FeatureList}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, feature_list)
        else
            PB.skip(d, wire_type)
        end
    end
    return FeatureLists(feature_list)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::FeatureLists)
    initpos = position(e.io)
    !isempty(x.feature_list) && PB.encode(e, 1, x.feature_list)
    return position(e.io) - initpos
end
function PB._encoded_size(x::FeatureLists)
    encoded_size = 0
    !isempty(x.feature_list) && (encoded_size += PB._encoded_size(x.feature_list, 1))
    return encoded_size
end

struct SequenceExample
    context::Union{Nothing,Features}
    feature_lists::Union{Nothing,FeatureLists}
end
PB.default_values(::Type{SequenceExample}) = (;context = nothing, feature_lists = nothing)
PB.field_numbers(::Type{SequenceExample}) = (;context = 1, feature_lists = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SequenceExample})
    context = Ref{Union{Nothing,Features}}(nothing)
    feature_lists = Ref{Union{Nothing,FeatureLists}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, context)
        elseif field_number == 2
            PB.decode!(d, feature_lists)
        else
            PB.skip(d, wire_type)
        end
    end
    return SequenceExample(context[], feature_lists[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SequenceExample)
    initpos = position(e.io)
    !isnothing(x.context) && PB.encode(e, 1, x.context)
    !isnothing(x.feature_lists) && PB.encode(e, 2, x.feature_lists)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SequenceExample)
    encoded_size = 0
    !isnothing(x.context) && (encoded_size += PB._encoded_size(x.context, 1))
    !isnothing(x.feature_lists) && (encoded_size += PB._encoded_size(x.feature_lists, 2))
    return encoded_size
end
end # module
