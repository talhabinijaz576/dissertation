# Generated by Django 3.2.5 on 2021-08-08 17:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portal', '0003_camera_account_owner'),
    ]

    operations = [
        migrations.AddField(
            model_name='camera',
            name='token',
            field=models.CharField(db_index=True, default='', max_length=100),
        ),
    ]
