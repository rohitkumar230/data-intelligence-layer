schema = {
    "brand_reach_by_date_distribution": """
|-- date: date (nullable = true)
|-- region: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
""",
    "brand_frequency_distribution": """
|-- tv_type: string (nullable = true)
|-- quintile: string (nullable = true)
|-- region: string (nullable = true)
|-- frequency: double (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- percent_imps: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_overview": """
|-- tv_type: string (nullable = true)
|-- region: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_demographic_quintile_distribution": """
|-- tv_type: string (nullable = true)
|-- region: string (nullable = true)
|-- quintile: string (nullable = true)
|-- segment_type: string (nullable = true)
|-- segment_name: string (nullable = true)
|-- impressions: long (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_geo_distribution": """
|-- tv_type: string (nullable = true)
|-- region: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- total_active_tvs: long (nullable = true)
|-- percentage_reach_hh: double (nullable = true)
|-- percentage_impression: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_yt_video_distribution": """
|-- region: string (nullable = true)
|-- video_name: string (nullable = true)
|-- total_views: long (nullable = true)
|-- paid_views: long (nullable = true)
|-- organic_views: long (nullable = true)
|-- spend_estimate: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- channel_name: string (nullable = true)
|-- channel_id: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_ads": """
|-- tv_type: string (nullable = true)
|-- region: string (nullable = true)
|-- ad_name: string (nullable = true)
|-- ad_duration: string (nullable = true)
|-- product: string (nullable = true)
|-- category: string (nullable = true)
|-- first_aired: date (nullable = true)
|-- last_aired: date (nullable = true)
|-- thumbnail_link: string (nullable = true)
|-- video_link: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_yt_overview": """
|-- region: string (nullable = true)
|-- views: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_linear_tv_entity_quintile_distribution": """
|-- region: string (nullable = true)
|-- quintile: string (nullable = true)
|-- entity_type: string (nullable = true)
|-- entity_name: string (nullable = true)
|-- impressions: long (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_ott_entity_quintile_distribution": """
|-- region: string (nullable = true)
|-- quintile: string (nullable = true)
|-- entity_type: string (nullable = true)
|-- entity_name: string (nullable = true)
|-- device_type: string (nullable = true)
|-- impressions: long (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_yt_channel_distribution": """
|-- region: string (nullable = true)
|-- channel_name: string (nullable = true)
|-- channel_id: string (nullable = true)
|-- total_views: long (nullable = true)
|-- paid_views: long (nullable = true)
|-- organic_views: long (nullable = true)
|-- spend_estimate: double (nullable = true)
|-- total_subscribers: long (nullable = true)
|-- no_of_videos: long (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_ott_entity_distribution": """
|-- region: string (nullable = true)
|-- entity_type: string (nullable = true)
|-- entity_name: string (nullable = true)
|-- device_type: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
    "brand_linear_tv_entity_distribution": """
|-- region: string (nullable = true)
|-- entity_type: string (nullable = true)
|-- entity_name: string (nullable = true)
|-- impressions: long (nullable = true)
|-- reach: long (nullable = true)
|-- frequency: double (nullable = true)
|-- brand_name: string (nullable = true)
|-- duration: string (nullable = true)
""",
}
